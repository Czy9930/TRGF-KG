import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


ROOT = Path(__file__).resolve().parent
SOURCE_KG = ROOT / 'data' / 'temporal_kg'
SPLIT_PATH = ROOT / 'data' / 'input' / 'typhoon_split_manifest.csv'
DEFAULT_OUTPUT = ROOT / 'data' / 'kg_enriched_v1'
BASE_RELATIONS = ('disaster', 'data_source', 'HAS_OBSERVATION', 'NEXT_OBS')
ATTRIBUTES = {
    'move_speed': ('MotionBin', 'HAS_MOVE_SPEED'),
    'radius7': ('Radius7Bin', 'HAS_RADIUS7'),
}
BUCKETS = 5


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def neo4j_credentials():
    user = os.environ.get('NEO4J_USER', 'neo4j')
    password = os.environ.get('NEO4J_' + 'PASSWORD')
    if not password:
        raise ValueError('请设置 NEO4J_PASSWORD 环境变量，脚本不会读取原导出器中的密码')
    return user, password


def fetch_obs(http_url, database):
    user, password = neo4j_credentials()
    query = ("MATCH (o:OBS) RETURN id(o) AS neo4j_id, "
             "o.tfbh AS tfbh, o.observation_index AS observation_index, "
             "o.move_speed AS move_speed, o.radius7 AS radius7")
    response = requests.post(
        f'{http_url.rstrip("/")}/db/{database}/tx/commit',
        json={'statements': [{'statement': query}]},
        auth=HTTPBasicAuth(user, password), timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get('errors'):
        raise RuntimeError(f'Neo4j 查询失败: {payload["errors"]}')
    result = payload['results'][0]
    columns = result['columns']
    frame = pd.DataFrame([item['row'] for item in result['data']],
                         columns=columns)
    if frame['neo4j_id'].duplicated().any():
        raise ValueError('Neo4j OBS id 存在重复')
    return frame


def fit_bins(mapping, train_ids):
    result = {}
    train = mapping.loc[mapping['typhoon_id'].isin(train_ids)]
    for name in ATTRIBUTES:
        values = pd.to_numeric(train[name], errors='coerce')
        values = values.loc[np.isfinite(values) & (values >= 0)]
        if len(values) < 100:
            raise ValueError(f'训练集 {name} 有效值不足')
        cuts = np.quantile(values.to_numpy(dtype=np.float64),
                           (0.25, 0.5, 0.75))
        if len(np.unique(cuts)) != 3:
            raise ValueError(f'{name} 四分位点重复，需要重新定义分箱')
        result[name] = cuts.tolist()
    return result


def assign_bins(mapping, thresholds):
    for name, cuts in thresholds.items():
        values = pd.to_numeric(mapping[name], errors='coerce')
        valid = np.isfinite(values) & (values >= 0)
        bucket = np.full(len(mapping), 4, dtype=np.int64)
        bucket[valid.to_numpy()] = np.searchsorted(
            cuts, values[valid].to_numpy(dtype=np.float64), side='right',
        )
        mapping[f'{name}_bin'] = bucket
    return mapping


def main(output_dir, http_url, database):
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f'输出目录已存在，避免覆盖: {output_dir}')
    base_meta = pd.read_csv(SOURCE_KG / 'entity_metadata.csv',
                            dtype={'typhoon_id': str, 'tfbh': str})
    mapping = pd.read_csv(SOURCE_KG / 'entity_time_mapping.csv',
                          dtype={'typhoon_id': str, 'tfbh': str})
    triples = pd.read_csv(SOURCE_KG / 'temporal_triples_full.csv',
                          encoding='utf-8-sig', dtype={'typhoon_id': str})
    split = pd.read_csv(SPLIT_PATH, encoding='utf-8-sig',
                        dtype={'typhoon_id': str})
    if (len(base_meta) != 27820 or len(mapping) != 26701
            or base_meta['entity_id'].duplicated().any()
            or mapping['entity_id'].duplicated().any()):
        raise ValueError('原始 KG 实体清单变化，请先审计映射')
    obs_meta = base_meta.loc[base_meta['node_type'] == 'Observation',
                             ['entity_id', 'neo4j_node_id']]
    mapping = mapping.merge(obs_meta, on='entity_id', validate='one_to_one')
    neo4j_obs = fetch_obs(http_url, database)
    mapping = mapping.merge(neo4j_obs, left_on='neo4j_node_id',
                            right_on='neo4j_id', validate='one_to_one',
                            suffixes=('', '_db'))
    if (len(mapping) != len(obs_meta)
            or not np.array_equal(mapping['tfbh'].astype(str).to_numpy(),
                                  mapping['tfbh_db'].astype(str).to_numpy())
            or not np.array_equal(mapping['time_step'].astype(int).to_numpy(),
                                  mapping['observation_index'].astype(int).to_numpy())):
        raise ValueError('Neo4j OBS 与原始轨迹实体的台风编号或时间步不一致')
    mapping['move_speed'] = mapping['move_speed_db']
    mapping['radius7'] = mapping['radius7']
    train_ids = set(split.loc[split['split'] == 'train', 'typhoon_id'])
    thresholds = fit_bins(mapping, train_ids)
    mapping = assign_bins(mapping, thresholds)
    meta_rows = []
    triple_rows = []
    next_id = int(base_meta['entity_id'].max()) + 1
    relation_names = list(BASE_RELATIONS)
    for name, (_, prefix) in ATTRIBUTES.items():
        relation_names += [f'{prefix}_Q{i}' for i in range(BUCKETS)]
    relation_ids = {name: i for i, name in enumerate(relation_names)}
    for typhoon_id, group in mapping.groupby('typhoon_id', sort=True):
        for name, (node_type, prefix) in ATTRIBUTES.items():
            buckets = group[f'{name}_bin'].astype(int)
            for bucket in sorted(set(buckets)):
                bin_id = next_id
                next_id += 1
                meta_rows.append({
                    'entity_id': bin_id,
                    'entity_name': f'{typhoon_id}_{name}_Q{bucket}',
                    'node_type': node_type,
                    'neo4j_node_id': np.nan,
                    'typhoon_id': typhoon_id,
                    'tfbh': np.nan,
                    'typhoon_name': np.nan,
                    'time_step': np.nan,
                })
                selected = group.loc[buckets == bucket]
                rel = f'{prefix}_Q{bucket}'
                for row in selected.itertuples(index=False):
                    triple_rows.append({
                        'head_id': int(row.entity_id), 'tail_id': bin_id,
                        'relation_id': relation_ids[rel], 'relation': rel,
                        'time_step': int(row.time_step),
                        'typhoon_id': typhoon_id,
                        'head_type': 'Observation', 'tail_type': node_type,
                    })
    new_meta = pd.concat([base_meta, pd.DataFrame(meta_rows)],
                         ignore_index=True)
    new_triples = pd.concat([triples, pd.DataFrame(triple_rows)],
                            ignore_index=True)
    if (new_meta['entity_id'].tolist() != list(range(next_id))
            or new_triples['relation_id'].max() != len(relation_names) - 1
            or new_triples[['head_id', 'tail_id']].max().max() >= next_id):
        raise ValueError('派生 KG 实体或关系 ID 不连续')
    for name in ATTRIBUTES:
        train_counts = mapping.loc[mapping['typhoon_id'].isin(train_ids),
                                   f'{name}_bin'].value_counts()
        if set(train_counts.index.astype(int)) != set(range(BUCKETS)):
            raise ValueError(f'训练集中 {name} 的分箱关系有空类')
    output_dir.mkdir(parents=True, exist_ok=False)
    new_meta.to_csv(output_dir / 'entity_metadata.csv', index=False)
    mapping.drop(columns=['neo4j_node_id', 'neo4j_id', 'tfbh_db',
                          'observation_index', 'move_speed_db']).to_csv(
        output_dir / 'entity_time_mapping.csv', index=False)
    new_triples.to_csv(output_dir / 'temporal_triples_full.csv',
                       index=False, encoding='utf-8-sig')
    with (output_dir / 'temporal_entities.txt').open('w', encoding='utf-8') as f:
        f.write(f'{len(new_meta)}\n')
        for row in new_meta.itertuples(index=False):
            f.write(f'{row.entity_name}\t{row.entity_id}\n')
    with (output_dir / 'temporal_relations.txt').open('w', encoding='utf-8') as f:
        f.write(f'{len(relation_names)}\n')
        for name, rel_id in relation_ids.items():
            f.write(f'{name}\t{rel_id}\n')
    config = {
        'source_kg': str(SOURCE_KG), 'source_kg_sha256': {
            name: sha256(SOURCE_KG / name)
            for name in ('entity_metadata.csv', 'entity_time_mapping.csv',
                         'temporal_triples_full.csv')},
        'split_manifest_sha256': sha256(SPLIT_PATH),
        'neo4j_database': database,
        'connection_is_read_only': True,
        'attributes': list(ATTRIBUTES),
        'training_only_quantile_thresholds': thresholds,
        'missing_bucket': 4,
        'relation_names': relation_names,
        'num_entities': len(new_meta),
        'num_triples': len(new_triples),
        'num_observations': len(mapping),
        'coverage': {name: int((mapping[f'{name}_bin'] != 4).sum())
                     for name in ATTRIBUTES},
        'train_bin_counts': {
            name: {str(k): int(v) for k, v in mapping.loc[
                mapping['typhoon_id'].isin(train_ids),
                f'{name}_bin'].value_counts().items()}
            for name in ATTRIBUTES},
    }
    with (output_dir / 'export_config.json').open('w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f'派生 KG 已导出: {output_dir}')
    print(f'实体 {len(new_meta)}, 原始关系 {len(relation_names)}, 三元组 {len(new_triples)}')
    print(f'训练集分箱阈值: {thresholds}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只读 Neo4j，导出派生属性关系 KG')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--neo4j-http', default='http://localhost:7474')
    parser.add_argument('--database', default='typhoon')
    args = parser.parse_args()
    main(args.output_dir, args.neo4j_http, args.database)
