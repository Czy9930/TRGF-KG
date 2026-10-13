import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
PERFORM = ROOT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_ORDER = ['lng', 'lat', 'speed', 'pressure']
KG_DIR = str(ROOT / 'data' / 'kg_enriched_v1')


LOOKBACK = 12
NODE_COUNT = 25
FEATURE_DIM = 10
ORIG_RELATIONS = 14
BATCH_SIZE = 128
DEFAULT_KG = ROOT / 'data' / 'kg_enriched_v1'
DEFAULT_INPUT = ROOT / 'data' / 'input'


class RGCNLayer(nn.Module):


    def __init__(self, in_dim, out_dim, num_relations, num_bases, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)
        self.bases = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.coefficients = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(self, x, edges_per_rel, deg_per_rel, num_nodes):
        weights = torch.einsum('rb,bij->rij', self.coefficients, self.bases)
        out = torch.zeros(num_nodes, self.out_dim, device=x.device)
        for rel_id in range(self.num_relations):
            src, dst = edges_per_rel[rel_id]
            if len(src) == 0:
                continue
            msg = x[src] @ weights[rel_id]
            rel_agg = torch.zeros(num_nodes, self.out_dim, device=x.device)
            rel_agg.index_add_(0, dst, msg)
            out += rel_agg / deg_per_rel[rel_id].unsqueeze(1)
        out = out + self.self_loop(x) + self.bias
        return self.dropout(out)


class RGCN(nn.Module):


    def __init__(self, in_dim, hidden_dim, out_dim, num_relations,
                 num_bases, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(RGCNLayer(
                in_dim, out_dim, num_relations, num_bases, dropout
            ))
        else:
            self.layers.append(RGCNLayer(
                in_dim, hidden_dim, num_relations, num_bases, dropout
            ))
            for _ in range(num_layers - 2):
                self.layers.append(RGCNLayer(
                    hidden_dim, hidden_dim, num_relations, num_bases, dropout
                ))
            self.layers.append(RGCNLayer(
                hidden_dim, out_dim, num_relations, num_bases, dropout
            ))
        self.skip_proj = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x, edges_per_rel, deg_per_rel, num_nodes):
        skip = self.skip_proj(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edges_per_rel, deg_per_rel, num_nodes)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return self.layer_norm(x + skip)


class RGCNLinkPredictor(nn.Module):


    def __init__(self, in_dim, hidden_dim, out_dim, num_relations,
                 num_bases, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = RGCN(in_dim, hidden_dim, out_dim, num_relations,
                            num_bases, num_layers, dropout)
        self.rel_emb = nn.Embedding(num_relations, out_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def encode(self, x, edges_per_rel, deg_per_rel, num_nodes):
        return self.encoder(x, edges_per_rel, deg_per_rel, num_nodes)

    def score(self, embeddings, head, rel, tail):
        return (
            embeddings[head] * self.rel_emb(rel) * embeddings[tail]
        ).sum(dim=-1)


def load_storm_structures(stats, processed_dir, kg_dir):
    triples = pd.read_csv(
        os.path.join(kg_dir, 'temporal_triples_full.csv'),
        encoding='utf-8-sig', dtype={'typhoon_id': str},
    )
    mapping = pd.read_csv(
        os.path.join(kg_dir, 'entity_time_mapping.csv'),
        encoding='utf-8-sig', dtype={'typhoon_id': str},
    )
    needed = set()
    for split in ('train', 'val', 'test'):
        with np.load(os.path.join(processed_dir, f'{split}_data.npz')) as z:
            needed.update(map(str, z['typhoon_ids']))
    triples = triples.loc[triples['typhoon_id'].isin(needed)]
    mapping = mapping.loc[mapping['typhoon_id'].isin(needed)]
    if set(triples['typhoon_id']) != needed or set(mapping['typhoon_id']) != needed:
        raise ValueError('forecasting 台风在新 KG 中缺失')

    structures = {}
    groups = {tid: group for tid, group in mapping.groupby('typhoon_id')}
    for tid, group in triples.groupby('typhoon_id'):
        static = {}
        for rel in ('disaster', 'data_source'):
            rows = group.loc[group['relation'] == rel]
            if len(rows) != 1:
                raise ValueError(f'{tid} 的 {rel} 边数量不是 1')
            static[rel] = (
                int(rows.iloc[0]['head_id']),
                int(rows.iloc[0]['tail_id']),
            )
        category_id, event_id = static['disaster']
        source_head, source_id = static['data_source']
        if event_id != source_head:
            raise ValueError(f'{tid} 的 Event→Source 路径断开')

        obs_rows = groups[tid].copy()
        if obs_rows['time_step'].duplicated().any():
            raise ValueError(f'{tid} 存在重复 OBS time_step')
        has_obs = group.loc[group['relation'] == 'HAS_OBSERVATION']
        if set(has_obs['tail_id']) != set(obs_rows['entity_id']):
            raise ValueError(f'{tid} 的 HAS_OBSERVATION 与 OBS 映射不一致')
        if not (has_obs['head_id'] == source_id).all():
            raise ValueError(f'{tid} 的 HAS_OBSERVATION 来源节点不一致')
        obs_id_set = set(obs_rows['entity_id'].astype(int))
        next_edges = group.loc[group['relation'] == 'NEXT_OBS']
        if not set(next_edges['head_id']).issubset(obs_id_set):
            raise ValueError(f'{tid} 的 NEXT_OBS head 不在 OBS 映射中')
        if not set(next_edges['tail_id']).issubset(obs_id_set):
            raise ValueError(f'{tid} 的 NEXT_OBS tail 不在 OBS 映射中')

        for col in FEATURE_ORDER:
            obs_rows[col] = pd.to_numeric(obs_rows[col], errors='coerce')
        obs_rows['speed'] = obs_rows['speed'].fillna(float(stats['speed_median']))
        obs_rows['pressure'] = obs_rows['pressure'].fillna(
            float(stats['pressure_median'])
        )
        values = obs_rows[FEATURE_ORDER].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'{tid} 的 OBS 数值特征无法归一化')
        normalized = (
            values - stats['feature_mean']
        ) / stats['feature_std']
        by_step = {}
        for row, features in zip(obs_rows.itertuples(index=False), normalized):
            one_obs = np.zeros(8, dtype=np.float32)
            one_obs[:4] = features
            one_obs[7] = 1.0
            by_step[int(row.time_step)] = (
                int(row.entity_id), one_obs
            )
        structures[tid] = {
            'category_id': category_id,
            'event_id': event_id,
            'source_id': source_id,
            'obs_by_step': by_step,
            'next_edges': list(zip(
                next_edges['head_id'].astype(int),
                next_edges['tail_id'].astype(int),
            )),
        }
    return structures


def load_model(model_dir, kg_dir, expected_seed=None):
    with (model_dir / 'rgcn_config.json').open(encoding='utf-8') as file:
        config = json.load(file)
    with (kg_dir / 'export_config.json').open(encoding='utf-8') as file:
        exported = json.load(file)
    if ((expected_seed is not None and config['seed'] != expected_seed)
            or config['num_orig_relations'] != 14
            or config['num_relations'] != 28
            or config['node_feature_dim'] != FEATURE_DIM
            or [row[0] for row in config['relation_schema']]
            != exported['relation_names']
            or Path(config['kg_input_dir']).resolve() != kg_dir.resolve()
            or config['negative_sampling'] !=
            'filtered_type_constrained_train_nodes'):
        raise ValueError('RGCN 检查点与当前派生 KG、seed 或训练约束不一致')
    with np.load(model_dir / 'kg_normalization_stats.npz',
                 allow_pickle=False) as file:
        stats = {key: file[key].copy() for key in file.files}
    if (stats['feature_mean'].shape != (4,)
            or stats['feature_std'].shape != (4,)):
        raise ValueError('OBS 训练期归一化参数维度错误')
    model = RGCNLinkPredictor(
        in_dim=FEATURE_DIM, hidden_dim=config['hidden_dim'],
        out_dim=config['embedding_dim'], num_relations=28,
        num_bases=config['num_bases'], num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(DEVICE)
    model.load_state_dict(torch.load(
        model_dir / 'rgcn_train_only.pth', map_location=DEVICE,
        weights_only=True,
    ))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config, stats


def load_structures(stats, kg_dir, processed_dir):

    structures = load_storm_structures(stats, str(processed_dir), kg_dir)
    mapping = pd.read_csv(kg_dir / 'entity_time_mapping.csv',
                          dtype={'typhoon_id': str})
    required = {'entity_id', 'typhoon_id', 'time_step',
                'move_speed_bin', 'radius7_bin'}
    if required - set(mapping.columns):
        raise ValueError(f'派生 OBS 映射缺少 {required - set(mapping.columns)}')
    groups = {tid: group for tid, group in
              mapping.groupby('typhoon_id', sort=False)}
    for tid, storm in structures.items():
        if tid not in groups:
            raise ValueError(f'派生图缺少台风 {tid}')
        bins = {}
        for row in groups[tid].itertuples(index=False):
            step = int(row.time_step)
            if (step in bins or step not in storm['obs_by_step']
                    or int(row.entity_id) != storm['obs_by_step'][step][0]):
                raise ValueError(f'{tid}: 属性分箱与原 OBS 时间映射不一致')
            motion, radius = int(row.move_speed_bin), int(row.radius7_bin)
            if not (0 <= motion < 5 and 0 <= radius < 5):
                raise ValueError(f'{tid}: 属性分箱越界')
            bins[step] = (motion, radius)
        if set(bins) != set(storm['obs_by_step']):
            raise ValueError(f'{tid}: 派生属性未覆盖全部 OBS')
        storm['bins_by_step'] = bins
    return structures


def build_sample(storm, input_steps, origin, changed_bin=None):
    steps = np.asarray(input_steps, dtype=np.int64)
    if (steps.shape != (LOOKBACK,) or len(set(map(int, steps))) != LOOKBACK
            or not np.all(np.diff(steps) > 0)
            or int(steps[-1]) != int(origin)):
        raise ValueError('历史窗口必须是 12 个递增时间步且末点等于预测起点')
    features = np.zeros((NODE_COUNT, FEATURE_DIM), dtype=np.float32)
    features[0, 4] = 1.0
    features[1, 5] = 1.0
    features[2, 6] = 1.0
    features[15:20, 8] = 1.0
    features[20:25, 9] = 1.0
    obs_local = {}
    edges = [[] for _ in range(ORIG_RELATIONS)]
    edges[0].append((0, 1))
    edges[1].append((1, 2))
    for i, step in enumerate(steps):
        step = int(step)
        if step not in storm['obs_by_step'] or step not in storm['bins_by_step']:
            raise ValueError(f'历史 OBS {step} 缺失')
        global_id, old_feature = storm['obs_by_step'][step]
        local_id = 3 + i
        features[local_id, :4] = old_feature[:4]
        features[local_id, 7] = 1.0
        obs_local[global_id] = local_id
        edges[2].append((2, local_id))
        motion, radius = storm['bins_by_step'][step]
        if changed_bin is not None and changed_bin[0] == step:
            motion = changed_bin[1]
        edges[4 + motion].append((local_id, 15 + motion))
        edges[9 + radius].append((local_id, 20 + radius))
    edges[3] = [
        (obs_local[head], obs_local[tail])
        for head, tail in storm['next_edges']
        if head in obs_local and tail in obs_local
    ]
    edges += [[(tail, head) for head, tail in relation]
              for relation in edges]
    if (len(edges) != 28
            or any(not (0 <= head < NODE_COUNT and 0 <= tail < NODE_COUNT)
                   for relation in edges for head, tail in relation)):
        raise ValueError('局部因果图的关系数量或端点异常')
    return features, edges


def encode_batch(model, samples):
    features = np.concatenate([sample[0] for sample in samples], axis=0)
    num_nodes = len(features)
    edge_map, degree_map = {}, {}
    for relation in range(28):
        pairs = [
            (head + i * NODE_COUNT, tail + i * NODE_COUNT)
            for i, (_, edges) in enumerate(samples)
            for head, tail in edges[relation]
        ]
        if pairs:
            src, dst = (torch.tensor([pair[j] for pair in pairs],
                                     dtype=torch.long, device=DEVICE)
                        for j in (0, 1))
        else:
            src = dst = torch.empty(0, dtype=torch.long, device=DEVICE)
        edge_map[relation] = (src, dst)
        degree = torch.zeros(num_nodes, device=DEVICE)
        degree.index_add_(0, dst, torch.ones(len(dst), device=DEVICE))
        degree_map[relation] = degree.clamp(min=1)
    with torch.no_grad():
        encoded = model.encode(torch.from_numpy(features).to(DEVICE),
                               edge_map, degree_map, num_nodes)
    return encoded.reshape(len(samples), NODE_COUNT, -1)[:, 3:15].cpu().numpy()


def check_causality(model, config, structures, processed_dir):
    rng = np.random.default_rng(20260929)
    result = {}
    for split in ('train', 'val', 'test'):
        with np.load(processed_dir / f'{split}_data.npz',
                     allow_pickle=False) as file:
            ids = file['typhoon_ids'].copy()
            input_steps = file['input_time_steps'].copy()
            future_steps = file['forecast_time_steps'].copy()
            origins = file['forecast_origin_time_steps'].copy()
        sampled_storms = rng.choice(np.unique(ids), size=20, replace=False)
        cases = []
        for tid in sampled_storms:
            index = int(rng.choice(np.flatnonzero(ids == tid)))
            storm = structures[str(tid)]
            steps = input_steps[index]
            future = int(rng.choice(future_steps[index]))
            history = int(rng.choice(steps))
            if future <= int(origins[index]) or future not in storm['bins_by_step']:
                raise ValueError('未来 OBS 时间步无效')
            original = encode_batch(model, [build_sample(
                storm, steps, origins[index])])[0]
            old_future_bin = storm['bins_by_step'][future][0]
            future_changed = encode_batch(model, [build_sample(
                storm, steps, origins[index],
                changed_bin=(future, (old_future_bin + 1) % 5))])[0]
            old_history_bin = storm['bins_by_step'][history][0]
            history_changed = encode_batch(model, [build_sample(
                storm, steps, origins[index],
                changed_bin=(history, (old_history_bin + 1) % 5))])[0]
            future_diff = float(np.max(np.abs(original - future_changed)))
            history_diff = float(np.max(np.abs(original - history_changed)))
            cases.append({'typhoon_id': str(tid), 'sample_index': index,
                          'future_difference': future_diff,
                          'historical_difference': history_diff,
                          'passed': future_diff == 0.0 and history_diff > 0.0})
        result[split] = {
            'passed': sum(item['passed'] for item in cases),
            'total': len(cases), 'cases': cases,
        }
        if result[split]['passed'] != len(cases):
            raise AssertionError(f'{split}: 因果敏感性检查未通过')
    return result


def generate_split(split, model, config, structures, processed_dir, output_dir):
    with np.load(processed_dir / f'{split}_data.npz',
                 allow_pickle=False) as file:
        ids = file['typhoon_ids'].copy()
        steps = file['input_time_steps'].copy()
        origins = file['forecast_origin_time_steps'].copy()
        starts = file['start_indices'].copy()
        years = file['years'].copy()
    count = len(ids)
    values = np.empty((count, LOOKBACK, config['embedding_dim']),
                      dtype=np.float32)
    for begin in range(0, count, BATCH_SIZE):
        end = min(begin + BATCH_SIZE, count)
        samples = [build_sample(structures[str(ids[i])], steps[i], origins[i])
                   for i in range(begin, end)]
        values[begin:end] = encode_batch(model, samples)
    if not np.isfinite(values).all():
        raise ValueError(f'{split}: 非有限嵌入')
    np.savez_compressed(output_dir / f'{split}_causal_embeddings.npz',
                        embeddings=values, typhoon_ids=ids,
                        input_time_steps=steps,
                        forecast_origin_time_steps=origins,
                        start_indices=starts, years=years)
    print(f'{split}: {values.shape}', flush=True)
    return {'samples': count, 'shape': list(values.shape)}


def main(model_dir, kg_dir, processed_dir, output_dir=None,
         expected_seed=None):
    model_dir, kg_dir, processed_dir = (
        path.resolve() for path in (model_dir, kg_dir, processed_dir)
    )
    model, config, stats = load_model(model_dir, kg_dir, expected_seed)
    output_dir = (Path(output_dir) if output_dir is not None else
                  ROOT / 'data' / 'causal_embeddings_v1' /
                  f'seed_{config["seed"]}').resolve()
    if output_dir.exists():
        raise FileExistsError(f'嵌入输出目录已存在: {output_dir}')
    structures = load_structures(stats, kg_dir, processed_dir)
    sensitivity = check_causality(
        model, config, structures, processed_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'causal_sensitivity_check.json').open(
            'w', encoding='utf-8') as file:
        json.dump(sensitivity, file, ensure_ascii=False, indent=2)
    summary = {
        'seed': config['seed'],
        'model_checkpoint': str(model_dir / 'rgcn_train_only.pth'),
        'processed_dir': str(processed_dir),
        'kg_dir': str(kg_dir),
        'num_storms': len(structures),
        'embedding_dim': config['embedding_dim'],
        'input_window': LOOKBACK,
        'graph_definition': 'historical-window-induced 25-node enriched causal subgraph',
        'graph_nodes_per_sample': NODE_COUNT,
        'causal_sensitivity_passed': True,
        'splits': {},
    }
    for split in ('train', 'val', 'test'):
        summary['splits'][split] = generate_split(
            split, model, config, structures, processed_dir, output_dir,
        )
    with (output_dir / 'causal_embedding_summary.json').open(
            'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f'完成 seed={config["seed"]}: {output_dir}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成派生属性 KG 因果嵌入')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--kg-dir', type=Path, default=DEFAULT_KG)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--expected-seed', type=int,
                        help='可选：核对检查点 seed；不填时从检查点读取')
    args = parser.parse_args()
    seed_hint = 42 if args.expected_seed is None else args.expected_seed
    model_dir = (args.model_dir or ROOT / 'result' / 'RGCN_enriched_v1' /
                 f'seed_{seed_hint}')
    main(model_dir, args.kg_dir, args.processed_dir,
         args.output_dir, args.expected_seed)
