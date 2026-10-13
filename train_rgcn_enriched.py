import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'kg_enriched_v1')
SPLIT_MANIFEST = os.path.join(
    SCRIPT_DIR, 'data', 'input', 'typhoon_split_manifest.csv'
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'result', 'RGCN_enriched_v1')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'hidden_dim': 128,
    'embedding_dim': 64,
    'num_bases': 8,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'batch_size': 8192,
    'negative_samples': 5,
    'margin': 1.0,
    'seed': 42,
    'relation_balance': 'inverse_frequency',
    'negative_sampling': 'filtered_type_constrained_train_nodes',
}

FEATURE_ORDER = ['lng', 'lat', 'speed', 'pressure']
NODE_TYPE_ORDER = ['Category', 'TyphoonEvent', 'Source', 'Observation', 'MotionBin', 'Radius7Bin']
RELATION_SCHEMA = [
    ('disaster', 'Category', 'TyphoonEvent'),
    ('data_source', 'TyphoonEvent', 'Source'),
    ('HAS_OBSERVATION', 'Source', 'Observation'),
    ('NEXT_OBS', 'Observation', 'Observation'),
    *((f'HAS_MOVE_SPEED_Q{i}', 'Observation', 'MotionBin') for i in range(5)),
    *((f'HAS_RADIUS7_Q{i}', 'Observation', 'Radius7Bin') for i in range(5)),
]


def load_split_manifest():
    df = pd.read_csv(SPLIT_MANIFEST, encoding='utf-8-sig',
                     dtype={'typhoon_id': str})
    required = {'typhoon_id', 'year', 'split'}
    if missing := required - set(df.columns):
        raise ValueError(f'划分清单缺少字段: {sorted(missing)}')
    if df['typhoon_id'].isna().any() or df['typhoon_id'].duplicated().any():
        raise ValueError('划分清单存在空或重复 typhoon_id')
    if not df['split'].isin(['train', 'val', 'test']).all():
        raise ValueError('划分清单存在未知 split')
    df['year'] = pd.to_numeric(df['year'], errors='raise').astype(int)
    for name, first, last in (
        ('train', 2000, 2018),
        ('val', 2019, 2020),
        ('test', 2021, 2022),
    ):
        rows = df.loc[df['split'] == name]
        if rows.empty or not rows['year'].between(first, last).all():
            raise ValueError(f'{name} 年份应在 {first}–{last}')
    print('台风划分:', df['split'].value_counts().to_dict())
    return df


def load_relation_schema(input_dir):
    with open(os.path.join(input_dir, 'temporal_relations.txt'),
              encoding='utf-8') as f:
        declared = int(f.readline().strip())
        mapping = {}
        for line in f:
            if line.strip():
                name, rel_id = line.strip().split('\t')
                mapping[name] = int(rel_id)
    expected = {name: i for i, (name, _, _) in enumerate(RELATION_SCHEMA)}
    if declared != len(expected) or mapping != expected:
        raise ValueError(f'KG 关系映射与预期十四种关系不一致: {mapping}')
    return mapping


def load_training_graph(input_dir):
    split = load_split_manifest()
    relation_ids = load_relation_schema(input_dir)
    with open(os.path.join(input_dir, 'temporal_entities.txt'),
              encoding='utf-8') as f:
        num_global_entities = int(f.readline().strip())

    metadata = pd.read_csv(
        os.path.join(input_dir, 'entity_metadata.csv'),
        dtype={'typhoon_id': str},
    )
    needed_meta = {'entity_id', 'node_type', 'typhoon_id'}
    if missing := needed_meta - set(metadata.columns):
        raise ValueError(f'实体元数据缺少字段: {sorted(missing)}')
    if metadata['entity_id'].duplicated().any():
        raise ValueError('实体元数据存在重复 ID')
    if len(metadata) != num_global_entities:
        raise ValueError('实体文件与元数据的数量不一致')
    if set(metadata['entity_id'].astype(int)) != set(range(num_global_entities)):
        raise ValueError('全局实体 ID 不是连续的 0..N-1')
    if not metadata['node_type'].isin(NODE_TYPE_ORDER).all():
        raise ValueError('KG 中存在未知节点类型')
    meta_by_id = metadata.set_index('entity_id')

    triples = pd.read_csv(
        os.path.join(input_dir, 'temporal_triples_full.csv'),
        encoding='utf-8-sig', dtype={'typhoon_id': str},
    )
    needed_triples = {
        'head_id', 'tail_id', 'relation_id', 'relation', 'time_step',
        'typhoon_id', 'head_type', 'tail_type',
    }
    if missing := needed_triples - set(triples.columns):
        raise ValueError(f'完整三元组缺少字段: {sorted(missing)}')
    if triples[list(needed_triples)].isna().any().any():
        raise ValueError('完整三元组存在空字段')
    if not triples['head_id'].between(0, num_global_entities - 1).all():
        raise ValueError('三元组 head_id 越界')
    if not triples['tail_id'].between(0, num_global_entities - 1).all():
        raise ValueError('三元组 tail_id 越界')
    expected_rel_ids = triples['relation'].map(relation_ids)
    if expected_rel_ids.isna().any() or not (
            expected_rel_ids.astype(int) == triples['relation_id']).all():
        raise ValueError('三元组 relation 与 relation_id 不一致')
    if set(split['typhoon_id']) - set(triples['typhoon_id']):
        raise ValueError('划分清单中的台风在新 KG 三元组中缺失')

    train_storms = set(split.loc[split['split'] == 'train', 'typhoon_id'])
    val_storms = set(split.loc[split['split'] == 'val', 'typhoon_id'])
    test_storms = set(split.loc[split['split'] == 'test', 'typhoon_id'])
    train_rows = triples.loc[triples['typhoon_id'].isin(train_storms)].copy()
    if train_rows.empty:
        raise ValueError('训练子图没有三元组')
    if set(train_rows['typhoon_id']) != train_storms:
        raise ValueError('有训练台风没有三元组')


    expected_types = {
        name: (head_type, tail_type)
        for name, head_type, tail_type in RELATION_SCHEMA
    }
    for row in train_rows.itertuples(index=False):
        head_meta = meta_by_id.loc[int(row.head_id)]
        tail_meta = meta_by_id.loc[int(row.tail_id)]
        expected_head, expected_tail = expected_types[row.relation]
        if (row.head_type, row.tail_type) != (
                expected_head, expected_tail):
            raise ValueError(f'{row.relation} 的 CSV 端点类型错误')
        if (head_meta['node_type'], tail_meta['node_type']) != (
                expected_head, expected_tail):
            raise ValueError(f'{row.relation} 的元数据端点类型错误')
        for endpoint in (head_meta, tail_meta):
            if endpoint['node_type'] != 'Category':
                if endpoint['typhoon_id'] != row.typhoon_id:
                    raise ValueError('训练边连接了其他台风的实体')

    train_node_ids = sorted(
        set(train_rows['head_id'].astype(int))
        | set(train_rows['tail_id'].astype(int))
    )
    nontrain_entities = set(
        metadata.loc[
            metadata['typhoon_id'].isin(val_storms | test_storms), 'entity_id'
        ].astype(int)
    )
    if set(train_node_ids) & nontrain_entities:
        raise ValueError('验证/测试实体进入训练子图')
    global_to_local = {
        global_id: local_id
        for local_id, global_id in enumerate(train_node_ids)
    }
    num_nodes = len(train_node_ids)
    num_orig_relations = len(RELATION_SCHEMA)
    forward = np.column_stack((
        train_rows['head_id'].map(global_to_local).to_numpy(dtype=np.int64),
        train_rows['tail_id'].map(global_to_local).to_numpy(dtype=np.int64),
        train_rows['relation_id'].to_numpy(dtype=np.int64),
    ))
    inverse = np.column_stack((
        forward[:, 1], forward[:, 0], forward[:, 2] + num_orig_relations
    ))

    train_triples = np.concatenate((forward, inverse), axis=0)
    num_relations = 2 * num_orig_relations

    obs_map = pd.read_csv(
        os.path.join(input_dir, 'entity_time_mapping.csv'),
        dtype={'typhoon_id': str},
    )
    needed_obs = {'entity_id', 'typhoon_id', *FEATURE_ORDER}
    if missing := needed_obs - set(obs_map.columns):
        raise ValueError(f'OBS 映射缺少字段: {sorted(missing)}')
    train_obs = obs_map.loc[obs_map['typhoon_id'].isin(train_storms)].copy()
    if train_obs['entity_id'].duplicated().any():
        raise ValueError('训练 OBS 存在重复实体 ID')
    train_obs_ids = set(train_obs['entity_id'].astype(int))
    if not train_obs_ids or not train_obs_ids.issubset(set(train_node_ids)):
        raise ValueError('训练 OBS 没有全部进入训练子图')
    if set(
        metadata.loc[
            metadata['node_type'] == 'Observation', 'entity_id'
        ].astype(int)
    ) & set(train_node_ids) != train_obs_ids:
        raise ValueError('训练图中 OBS 与训练台风 OBS 清单不一致')

    for col in FEATURE_ORDER:
        train_obs[col] = pd.to_numeric(train_obs[col], errors='coerce')
    if train_obs[['lng', 'lat']].isna().any().any():
        raise ValueError('训练 OBS 的经纬度存在缺失')
    speed_median = float(train_obs['speed'].median())
    pressure_median = float(train_obs['pressure'].median())
    if not np.isfinite([speed_median, pressure_median]).all():
        raise ValueError('训练 OBS 无法计算强度特征中位数')
    train_obs['speed'] = train_obs['speed'].fillna(speed_median)
    train_obs['pressure'] = train_obs['pressure'].fillna(pressure_median)
    obs_values = train_obs[FEATURE_ORDER].to_numpy(dtype=np.float32)
    if not np.isfinite(obs_values).all():
        raise ValueError('训练 OBS 含非有限数值')
    feature_mean = obs_values.mean(axis=0)
    feature_std = obs_values.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0


    node_features = np.zeros(
        (num_nodes, len(FEATURE_ORDER) + len(NODE_TYPE_ORDER)),
        dtype=np.float32,
    )
    node_types = []
    node_type_to_id = {name: i for i, name in enumerate(NODE_TYPE_ORDER)}
    type_pools_np = {}
    positions_np = np.empty(num_nodes, dtype=np.int64)
    for local_id, global_id in enumerate(train_node_ids):
        node_type = meta_by_id.loc[global_id, 'node_type']
        node_types.append(node_type_to_id[node_type])
        node_features[local_id, len(FEATURE_ORDER)
                      + node_type_to_id[node_type]] = 1.0
    for type_id, name in enumerate(NODE_TYPE_ORDER):
        pool = np.flatnonzero(np.array(node_types) == type_id)
        if len(pool) == 0:
            raise ValueError(f'训练图缺少 {name} 节点')
        type_pools_np[type_id] = pool
        positions_np[pool] = np.arange(len(pool))
    local_obs = np.array(
        [global_to_local[int(gid)] for gid in train_obs['entity_id']],
        dtype=np.int64,
    )
    node_features[local_obs, :len(FEATURE_ORDER)] = (
        obs_values - feature_mean
    ) / feature_std

    head_type_ids = []
    tail_type_ids = []
    for _, head_type, tail_type in RELATION_SCHEMA:
        head_type_ids.append(node_type_to_id[head_type])
        tail_type_ids.append(node_type_to_id[tail_type])
    head_type_ids += tail_type_ids.copy()
    tail_type_ids += head_type_ids[:num_orig_relations]
    node_type_array = np.array(node_types)
    for h, t, r in train_triples:
        if (node_type_array[h] != head_type_ids[r]
                or node_type_array[t] != tail_type_ids[r]):
            raise ValueError('训练图中的边违反关系端点类型约束')

    edges_per_rel = {}
    deg_per_rel = {}
    for rel in range(num_relations):
        relation_edges = train_triples[train_triples[:, 2] == rel]
        src = torch.as_tensor(relation_edges[:, 0], dtype=torch.long,
                              device=DEVICE)
        dst = torch.as_tensor(relation_edges[:, 1], dtype=torch.long,
                              device=DEVICE)
        edges_per_rel[rel] = (src, dst)
        degree = torch.zeros(num_nodes, device=DEVICE)
        degree.index_add_(0, dst, torch.ones(len(dst), device=DEVICE))
        deg_per_rel[rel] = degree.clamp(min=1)
    relation_counts = np.bincount(
        train_triples[:, 2], minlength=num_relations
    )
    if (relation_counts == 0).any():
        raise ValueError('有关系类型在训练图中没有正样本')


    heads_by_relation_tail = {}
    tails_by_head_relation = {}
    for h, t, r in train_triples:
        heads_by_relation_tail.setdefault((int(r), int(t)), set()).add(int(h))
        tails_by_head_relation.setdefault((int(h), int(r)), set()).add(int(t))
    can_replace_head = np.array([
        len(heads_by_relation_tail[(int(r), int(t))])
        < len(type_pools_np[head_type_ids[int(r)]])
        for h, t, r in train_triples
    ], dtype=np.bool_)
    can_replace_tail = np.array([
        len(tails_by_head_relation[(int(h), int(r))])
        < len(type_pools_np[tail_type_ids[int(r)]])
        for h, t, r in train_triples
    ], dtype=np.bool_)
    trainable_mask = can_replace_head | can_replace_tail
    if not trainable_mask.any():
        raise ValueError('所有训练三元组都没有合法的 filtered typed negative')
    trainable_triples = train_triples[trainable_mask]
    trainable_counts = np.bincount(
        trainable_triples[:, 2], minlength=num_relations
    )
    active_relations = trainable_counts > 0


    relation_weights = np.zeros(num_relations, dtype=np.float32)
    relation_weights[active_relations] = (
        len(trainable_triples)
        / (active_relations.sum() * trainable_counts[active_relations])
    )
    positive_keys_np = (
        (train_triples[:, 0] * num_relations + train_triples[:, 2])
        * num_nodes + train_triples[:, 1]
    )
    positive_keys = torch.from_numpy(
        np.unique(positive_keys_np).astype(np.int64)
    ).to(DEVICE)
    relation_names = (
        [name for name, _, _ in RELATION_SCHEMA]
        + [f'{name}_inverse' for name, _, _ in RELATION_SCHEMA]
    )
    skipped_relations = [
        relation_names[i] for i in range(num_relations)
        if not active_relations[i]
    ]

    node_counts = {
        name: int(len(type_pools_np[i]))
        for i, name in enumerate(NODE_TYPE_ORDER)
    }
    counts_by_relation = {
        name: int(relation_counts[i])
        for i, (name, _, _) in enumerate(RELATION_SCHEMA)
    }
    print(f'Train-only 节点: {num_nodes}; '
          f'原始边: {len(forward)}; 含逆边: {len(train_triples)}')
    print('节点类型:', node_counts)
    print('原始关系边数:', counts_by_relation)
    print('归一化: 仅训练 OBS 拟合；负采样: 训练节点内按类型采样并过滤正边')
    print(f'DistMult 跳过无合法负样本的关系: {skipped_relations}')

    return {
        'node_features': torch.from_numpy(node_features).to(DEVICE),
        'num_nodes': num_nodes,
        'feature_dim': node_features.shape[1],
        'num_relations': num_relations,
        'num_orig_relations': num_orig_relations,
        'edges_per_rel': edges_per_rel,
        'deg_per_rel': deg_per_rel,
        'triples': torch.from_numpy(trainable_triples),
        'positive_keys': positive_keys,
        'can_replace_head': torch.from_numpy(can_replace_head[trainable_mask]),
        'can_replace_tail': torch.from_numpy(can_replace_tail[trainable_mask]),
        'relation_weights': torch.from_numpy(
            relation_weights
        ).to(DEVICE),
        'type_pools': {
            type_id: torch.from_numpy(pool).to(DEVICE)
            for type_id, pool in type_pools_np.items()
        },
        'node_pool_positions': torch.from_numpy(positions_np).to(DEVICE),
        'head_type_ids': head_type_ids,
        'tail_type_ids': tail_type_ids,
        'train_node_ids': np.array(train_node_ids, dtype=np.int64),
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'speed_median': speed_median,
        'pressure_median': pressure_median,
        'summary': {
            'train_storms': len(train_storms),
            'val_storms': len(val_storms),
            'test_storms': len(test_storms),
            'raw_kg_storms': int(triples['typhoon_id'].nunique()),
            'train_obs': len(train_obs_ids),
            'train_nodes': num_nodes,
            'train_forward_triples': len(forward),
            'train_triples_with_inverse': len(train_triples),
            'distmult_training_triples': len(trainable_triples),
            'distmult_relation_counts': trainable_counts.tolist(),
            'relations_without_filtered_negatives': skipped_relations,
            'node_type_counts': node_counts,
            'forward_relation_counts': counts_by_relation,
            'all_relation_counts': relation_counts.tolist(),
            'relation_loss_weights': relation_weights.tolist(),
        },
    }


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


def sample_typed_negatives(head, tail, relation, graph_data,
                           can_replace_head, can_replace_tail):

    if not torch.all(can_replace_head | can_replace_tail):
        raise ValueError('待采样正边不存在合法的 filtered typed negative')
    neg_head = head.clone()
    neg_tail = tail.clone()
    choose_head = torch.rand(len(head), device=DEVICE) < 0.5
    choose_head[~can_replace_head] = False
    choose_head[~can_replace_tail] = True
    pools = graph_data['type_pools']
    positions = graph_data['node_pool_positions']
    positive_keys = graph_data['positive_keys']
    num_nodes = graph_data['num_nodes']
    num_relations = graph_data['num_relations']
    retry = torch.ones(len(head), dtype=torch.bool, device=DEVICE)
    for _ in range(100):
        for rel_id in range(num_relations):
            rel_mask = (relation == rel_id) & retry
            if not rel_mask.any():
                continue
            for mask, original, pool, result in (
                (rel_mask & choose_head, head,
                 pools[graph_data['head_type_ids'][rel_id]], neg_head),
                (rel_mask & ~choose_head, tail,
                 pools[graph_data['tail_type_ids'][rel_id]], neg_tail),
            ):
                if not mask.any():
                    continue
                if len(pool) <= 1:
                    raise ValueError('选中的替换端没有足够同类型训练节点')
                current = original[mask]
                draw = torch.randint(
                    len(pool) - 1, (len(current),), device=DEVICE
                )
                draw += (draw >= positions[current]).long()
                result[mask] = pool[draw]

        candidate_keys = (
            (neg_head * num_relations + relation) * num_nodes + neg_tail
        )
        positions_in_sorted = torch.searchsorted(positive_keys, candidate_keys)
        safe_positions = positions_in_sorted.clamp(max=len(positive_keys) - 1)
        retry = ((positions_in_sorted < len(positive_keys))
                 & (positive_keys[safe_positions] == candidate_keys))
        if not retry.any():
            return neg_head, neg_tail
    raise RuntimeError('100 次重采样后仍碰到真实正三元组')


def train_rgcn(model, graph_data, config):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    triples = graph_data['triples']
    relation_weights = graph_data['relation_weights']
    losses = []
    print(f'开始预训练: device={DEVICE}, 固定 {config["epochs"]} epochs')
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        permutation = torch.randperm(len(triples))
        weighted_loss_sum = 0.0
        sample_count = 0
        for offset in range(0, len(triples), config['batch_size']):
            batch_indices = permutation[offset:offset + config['batch_size']]
            batch = triples[batch_indices].to(DEVICE)
            can_head = graph_data['can_replace_head'][batch_indices].to(DEVICE)
            can_tail = graph_data['can_replace_tail'][batch_indices].to(DEVICE)
            head, tail, relation = batch[:, 0], batch[:, 1], batch[:, 2]
            embeddings = model.encode(
                graph_data['node_features'],
                graph_data['edges_per_rel'],
                graph_data['deg_per_rel'],
                graph_data['num_nodes'],
            )
            positive = model.score(embeddings, head, relation, tail)
            negative_scores = []
            for _ in range(config['negative_samples']):
                neg_head, neg_tail = sample_typed_negatives(
                    head, tail, relation, graph_data, can_head, can_tail
                )
                negative_scores.append(
                    model.score(embeddings, neg_head, relation, neg_tail)
                )
            negatives = torch.stack(negative_scores, dim=1)
            per_positive_loss = F.relu(
                config['margin'] - positive[:, None] + negatives
            ).mean(dim=1)
            loss = (
                per_positive_loss * relation_weights[relation]
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            weighted_loss_sum += loss.item() * len(batch)
            sample_count += len(batch)
        epoch_loss = weighted_loss_sum / sample_count
        losses.append(epoch_loss)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch:3d}/{config["epochs"]} | '
                  f'balanced loss: {epoch_loss:.4f}', flush=True)
    return losses


def main(seed=None, output_dir=None, kg_dir=None, epochs=None, validate_only=False):
    print('Train-only 多关系 R-GCN + DistMult 预训练')
    config = {**CONFIG, 'seed': CONFIG['seed'] if seed is None else int(seed)}
    if epochs is not None:
        config['epochs'] = int(epochs)
    global INPUT_DIR
    INPUT_DIR = INPUT_DIR if kg_dir is None else os.path.abspath(kg_dir)
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, f'seed_{config["seed"]}')
    output_dir = os.path.abspath(output_dir)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with open(os.path.join(INPUT_DIR, 'export_config.json'), encoding='utf-8') as file:
        exported = json.load(file)
    if exported['relation_names'] != [name for name, _, _ in RELATION_SCHEMA]:
        raise ValueError('派生 KG 关系 schema 不一致')
    graph_data = load_training_graph(INPUT_DIR)
    if validate_only:
        print(f'训练图审计通过: {graph_data["num_nodes"]} 节点, '
              f'{graph_data["num_orig_relations"]} 正向关系, '
              f'{graph_data["feature_dim"]} 维节点特征')
        return
    if os.path.exists(output_dir):
        raise FileExistsError(f'RGCN 输出目录已存在: {output_dir}')
    os.makedirs(output_dir, exist_ok=False)
    model = RGCNLinkPredictor(
        in_dim=graph_data['feature_dim'],
        hidden_dim=config['hidden_dim'],
        out_dim=config['embedding_dim'],
        num_relations=graph_data['num_relations'],
        num_bases=config['num_bases'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(DEVICE)
    start = time.time()
    losses = train_rgcn(model, graph_data, config)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.save(
        model.state_dict(), os.path.join(output_dir, 'rgcn_train_only.pth')
    )
    np.savez(
        os.path.join(output_dir, 'kg_normalization_stats.npz'),
        feature_mean=graph_data['feature_mean'],
        feature_std=graph_data['feature_std'],
        speed_median=graph_data['speed_median'],
        pressure_median=graph_data['pressure_median'],
    )
    np.save(
        os.path.join(output_dir, 'train_node_global_ids.npy'),
        graph_data['train_node_ids'],
    )
    config_to_save = {
        **config,
        'feature_order': FEATURE_ORDER,
        'node_type_order': NODE_TYPE_ORDER,
        'relation_schema': RELATION_SCHEMA,
        'num_orig_relations': graph_data['num_orig_relations'],
        'num_relations': graph_data['num_relations'],
        'node_feature_dim': graph_data['feature_dim'],
        'kg_input_dir': INPUT_DIR,
        'split_manifest': SPLIT_MANIFEST,
    }
    with open(os.path.join(output_dir, 'rgcn_config.json'),
              'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)
    summary = {
        **graph_data['summary'],
        'seed': config['seed'],
        'epochs_completed': len(losses),
        'first_train_loss': losses[0],
        'final_train_loss': losses[-1],
        'training_seconds': round(time.time() - start, 2),
        'device': str(DEVICE),
    }
    with open(os.path.join(output_dir, 'training_summary.json'),
              'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'预训练完成: {len(losses)} epochs, '
          f'final balanced loss={losses[-1]:.4f}')
    print(f'模型及训练统计量: {output_dir}')
    print('下一步需按 input_time_steps 构造 G_t^hist 后生成因果嵌入。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='独立训练派生属性关系 R-GCN')
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--kg-dir', default=INPUT_DIR)
    parser.add_argument('--output-dir')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    main(args.seed, args.output_dir, args.kg_dir, args.epochs,
         args.validate_only)
