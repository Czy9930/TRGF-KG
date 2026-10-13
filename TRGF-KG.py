import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'input')
OUTPUT_PATH = os.path.join(
    SCRIPT_DIR, 'data', 'reference_forecasts', 'test_references.npz'
)
EARTH_RADIUS_KM = 6371.0088
HOUR = np.timedelta64(1, 'h')
LOOKBACK = 12
HORIZON = 24
EXPECTED_SPLIT_SAMPLES = {'train': 7527, 'test': 1367}
LONGITUDE_GRID = np.arange(30.0, 220.1, 2.0)
LATITUDE_GRID = np.arange(0.0, 70.1, 2.0)

TAU_U_H = 70.0
TAU_V_H = 46.0
BARNES_PASSES = ((6250.0, 3375.0), (1875.0, 1250.0))
ALIGNMENT_KEYS = (
    'typhoon_ids', 'start_indices', 'input_time_steps',
    'forecast_time_steps', 'forecast_origin_time_steps', 'years',
    'input_times', 'forecast_times', 'forecast_origin_time',
    'forecast_elapsed_hours',
)


def wrapped_longitude_difference(left, right):

    return (left - right + np.pi) % (2 * np.pi) - np.pi


def motion_kmh(lon0, lat0, lon1, lat1, hours):
    if hours <= 0:
        raise ValueError('计算台风运动必须使用正的真实时间间隔')
    lon0, lat0, lon1, lat1 = np.deg2rad([lon0, lat0, lon1, lat1])
    zonal = (EARTH_RADIUS_KM * np.cos((lat0 + lat1) / 2)
             * wrapped_longitude_difference(lon1, lon0) / hours)
    meridional = EARTH_RADIUS_KM * (lat1 - lat0) / hours
    return float(zonal), float(meridional)


def reference_load_split(split, metadata, input_dir=INPUT_DIR):
    with np.load(os.path.join(input_dir, f'{split}_data.npz'),
                 allow_pickle=False) as file:
        missing = {'X', 'y', *ALIGNMENT_KEYS} - set(file.files)
        if missing:
            raise ValueError(f'{split} 缺少字段: {sorted(missing)}')
        data = {key: file[key].copy() for key in ('X', 'y', *ALIGNMENT_KEYS)}
    n = len(data['X'])
    if (n != EXPECTED_SPLIT_SAMPLES[split]
            or data['X'].shape != (n, LOOKBACK, 4)
            or data['y'].shape != (n, HORIZON, 2)):
        raise ValueError(f'{split} 的轨迹/目标维度不符合 12×4 → 24×2')
    if (data['input_times'].shape != (n, LOOKBACK)
            or data['forecast_times'].shape != (n, HORIZON)
            or data['forecast_elapsed_hours'].shape != (n, HORIZON)
            or not np.array_equal(data['input_time_steps'][:, -1],
                                  data['forecast_origin_time_steps'])
            or not np.all(np.diff(data['input_time_steps'], axis=1) > 0)
            or not np.all(np.diff(data['forecast_time_steps'], axis=1) > 0)
            or not np.all(data['forecast_time_steps'][:, 0]
                              > data['forecast_origin_time_steps'])
            or np.isnat(data['input_times']).any()
            or np.isnat(data['forecast_times']).any()
            or np.isnat(data['forecast_origin_time']).any()
            or not np.array_equal(
                data['input_times'][:, -1], data['forecast_origin_time'])
            or not np.all(np.diff(data['input_times'], axis=1) > np.timedelta64(0, 'ns'))
            or not np.all(np.diff(data['forecast_times'], axis=1) > np.timedelta64(0, 'ns'))
            or not np.all(data['forecast_times'][:, 0] > data['forecast_origin_time'])
            or not np.allclose(
                data['forecast_elapsed_hours'],
                (data['forecast_times'] - data['forecast_origin_time'][:, None]) / HOUR,
            )):
        raise ValueError(f'{split} 的观测时间或真实提前时间无效')
    data['input_coordinates'] = metadata['scaler_X'].inverse_transform(
        data['X'].reshape(-1, 4)
    ).reshape(n, LOOKBACK, 4)[..., :2]
    data['ground_truth'] = metadata['scaler_y'].inverse_transform(
        data['y'].reshape(-1, 2)
    ).reshape(n, HORIZON, 2)
    if (not np.isfinite(data['X']).all()
            or not np.isfinite(data['y']).all()
            or not np.isfinite(data['input_coordinates']).all()
            or not np.isfinite(data['ground_truth']).all()):
        raise ValueError(f'{split} 的原始尺度经纬度包含非有限值')
    return data


def unique_training_observations(train):

    if not np.all((train['years'] >= 2000) & (train['years'] <= 2018)):
        raise ValueError('气候运动场只能使用 2000–2018 年训练台风')
    storms = {}
    step_times = {}
    for i, typhoon_id in enumerate(train['typhoon_ids']):
        tid = str(typhoon_id)
        pieces = (
            (train['input_time_steps'][i], train['input_times'][i],
             train['input_coordinates'][i]),
            (train['forecast_time_steps'][i], train['forecast_times'][i],
             train['ground_truth'][i]),
        )
        for steps, times, positions in pieces:
            for step, stamp, position in zip(steps, times, positions):
                timestamp_ns = int(stamp.astype('datetime64[ns]').astype(np.int64))
                key = (tid, timestamp_ns)
                step_key = (tid, int(step))
                if step_key in step_times and step_times[step_key] != timestamp_ns:
                    raise ValueError(f'{tid} OBS_{step} 对应多个原始时间')
                step_times[step_key] = timestamp_ns
                records = storms.setdefault(tid, {})
                coordinate = np.asarray(position, dtype=np.float64)
                if key in records:
                    prior_step, prior_coordinate = records[key]
                    if (prior_step != int(step)
                            or not np.allclose(prior_coordinate, coordinate,
                                               rtol=0, atol=1e-4)):
                        raise ValueError(f'{tid} 同一观测在不同窗口中不一致')
                else:
                    records[key] = (int(step), coordinate)
    velocities = []
    velocity_times = []
    for tid, records in storms.items():
        ordered = sorted((stamp, item[1]) for (storm, stamp), item
                         in records.items() if storm == tid)
        for (old_time, old_pos), (new_time, new_pos) in zip(ordered, ordered[1:]):
            hours = (new_time - old_time) / 3_600_000_000_000
            u, v = motion_kmh(*old_pos, *new_pos, hours)
            latitude = float(new_pos[1])
            longitude = float(new_pos[0] % 360)
            if (0 <= latitude <= 70 and 30 <= longitude <= 220):
                velocities.append((longitude, latitude, u, v))
                velocity_times.append(new_time)
    if not velocities:
        raise ValueError('训练期没有 J-North 范围内的气候运动观测')
    return (np.asarray(velocities, dtype=np.float64),
            np.asarray(velocity_times, dtype='datetime64[ns]'),
            len(storms), len(step_times))


def haversine_grid_distance_km(lon_a, lat_a, lon_b, lat_b):
    lon_a, lat_a, lon_b, lat_b = map(np.deg2rad, (lon_a, lat_a, lon_b, lat_b))
    dlon = wrapped_longitude_difference(lon_a, lon_b)
    dlat = lat_a - lat_b
    value = (np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b)
             * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def barnes_analysis(query_lon, query_lat, observations, values,
                    influence_km, efolding_km, fallback):

    result = np.empty((len(query_lon), 2), dtype=np.float64)
    for start in range(0, len(query_lon), 128):
        end = min(start + 128, len(query_lon))
        distance = haversine_grid_distance_km(
            query_lon[start:end, None], query_lat[start:end, None],
            observations[None, :, 0], observations[None, :, 1],
        )
        weight = np.exp(-(distance / efolding_km) ** 2)
        weight[distance > influence_km] = 0.0
        denominator = weight.sum(axis=1)
        block = result[start:end]
        block[:] = fallback
        valid = denominator > 0
        block[valid] = (
            weight[valid] @ values / denominator[valid, None]
        )
    return result


def bilinear(field, longitude, latitude):
    longitude = np.clip(np.asarray(longitude), LONGITUDE_GRID[0], LONGITUDE_GRID[-1])
    latitude = np.clip(np.asarray(latitude), LATITUDE_GRID[0], LATITUDE_GRID[-1])
    ix = np.clip(np.searchsorted(LONGITUDE_GRID, longitude, side='right') - 1,
                 0, len(LONGITUDE_GRID) - 2)
    iy = np.clip(np.searchsorted(LATITUDE_GRID, latitude, side='right') - 1,
                 0, len(LATITUDE_GRID) - 2)
    fx = (longitude - LONGITUDE_GRID[ix]) / (LONGITUDE_GRID[ix + 1] - LONGITUDE_GRID[ix])
    fy = (latitude - LATITUDE_GRID[iy]) / (LATITUDE_GRID[iy + 1] - LATITUDE_GRID[iy])
    return ((1 - fx)[..., None] * (1 - fy)[..., None] * field[iy, ix]
            + fx[..., None] * (1 - fy)[..., None] * field[iy, ix + 1]
            + (1 - fx)[..., None] * fy[..., None] * field[iy + 1, ix]
            + fx[..., None] * fy[..., None] * field[iy + 1, ix + 1])


def build_climatology(velocities, velocity_times):

    grid_lon, grid_lat = np.meshgrid(LONGITUDE_GRID, LATITUDE_GRID)
    locations = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))
    months = velocity_times.astype('datetime64[M]').astype(np.int64) % 12 + 1
    fields = np.empty((12, len(LATITUDE_GRID), len(LONGITUDE_GRID), 2))
    monthly_counts = {}
    for month in range(1, 13):
        current = velocities[months == month]
        if len(current) == 0:
            raise ValueError(f'训练期 {month} 月没有气候运动样本，不能生成完整气候场')
        monthly_counts[str(month)] = int(len(current))
        mean = current[:, 2:4].mean(axis=0)
        background = barnes_analysis(
            locations[:, 0], locations[:, 1], current[:, :2], current[:, 2:4],
            *BARNES_PASSES[0], mean,
        ).reshape(len(LATITUDE_GRID), len(LONGITUDE_GRID), 2)
        residual = current[:, 2:4] - bilinear(
            background, current[:, 0], current[:, 1],
        )
        correction = barnes_analysis(
            locations[:, 0], locations[:, 1], current[:, :2], residual,
            *BARNES_PASSES[1], np.zeros(2),
        ).reshape(background.shape)
        fields[month - 1] = background + correction
        print(f'Barnes 月份 {month:02d}/12：{len(current)} 个去重运动观测', flush=True)
    return fields, monthly_counts


def month_interpolation(date):
    middle = datetime(date.year, date.month, 15)
    if date >= middle:
        left = middle
        right = datetime(date.year + (date.month == 12), date.month % 12 + 1, 15)
    else:
        right = middle
        left = datetime(date.year - (date.month == 1),
                        12 if date.month == 1 else date.month - 1, 15)
    fraction = (date - left).total_seconds() / (right - left).total_seconds()
    return left.month - 1, right.month - 1, fraction


def climate_motion(fields, longitude_rad, latitude_rad, date):
    left, right, fraction = month_interpolation(date)
    lon = np.rad2deg(longitude_rad) % 360
    lat = np.rad2deg(latitude_rad)
    return ((1 - fraction) * bilinear(fields[left], lon, lat)
            + fraction * bilinear(fields[right], lon, lat))


def initial_motion(inputs, times, steps):
    origin_time = times[-1]
    lags = (origin_time - times[:-1]) / HOUR
    if not np.isfinite(lags).all() or not np.all(lags > 0):
        raise ValueError('历史时间必须严格早于 forecast origin')
    position = int(np.argmin(np.abs(lags - 12.0)))
    u, v = motion_kmh(*inputs[position], *inputs[-1], float(lags[position]))
    return np.array([u, v]), int(steps[position]), float(lags[position])


def integrate(initial_coordinates, velocity, origin_time, lead_hours, fields):

    origin = origin_time.astype('datetime64[us]').astype(datetime)
    results = np.empty((len(lead_hours), 2), dtype=np.float64)
    state = np.deg2rad(np.asarray(initial_coordinates, dtype=np.float64))
    previous_derivative = None
    previous_step = None
    elapsed = 0.0

    def derivative(at_state, at_time):
        u, v = velocity
        if fields is not None:
            climate_u, climate_v = climate_motion(
                fields, at_state[0], at_state[1],
                origin + timedelta(hours=at_time),
            )
            weight_u = np.exp(-at_time / TAU_U_H)
            weight_v = np.exp(-at_time / TAU_V_H)
            u = weight_u * u + (1 - weight_u) * climate_u
            v = weight_v * v + (1 - weight_v) * climate_v
        cosine = max(np.cos(at_state[1]), 1e-5)
        return np.array([u / (EARTH_RADIUS_KM * cosine),
                         v / EARTH_RADIUS_KM])

    for index, target in enumerate(lead_hours):
        if not np.isfinite(target) or target <= elapsed:
            raise ValueError('预测位置的真实提前时间必须严格递增')
        while elapsed < target - 1e-10:
            step = min(1.0, float(target - elapsed))
            current_derivative = derivative(state, elapsed)
            if previous_derivative is None:
                next_state = state + step * current_derivative
            else:
                ratio = step / (2 * previous_step)
                next_state = state + step * (
                    (1 + ratio) * current_derivative - ratio * previous_derivative
                )
            next_state[1] = np.clip(next_state[1], np.deg2rad(-89.5),
                                    np.deg2rad(89.5))
            previous_derivative = current_derivative
            previous_step = step
            state = next_state
            elapsed = min(float(target), elapsed + step)
        results[index] = (np.rad2deg(state[0]) % 360,
                          np.rad2deg(state[1]))
    return results


persistence_method = SimpleNamespace(
    EARTH_RADIUS_KM=EARTH_RADIUS_KM,
    wrapped_longitude_difference=wrapped_longitude_difference,
    load_split=reference_load_split,
    unique_training_observations=unique_training_observations,
    build_climatology=build_climatology,
    initial_motion=initial_motion,
    integrate=integrate,
)

if hasattr(__import__('sys').stdout, 'reconfigure'):
    __import__('sys').stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJECTORY_DIR = os.path.join(SCRIPT_DIR, 'data', 'input')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')
OUTPUT_DIR = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = (68, 4321, 2024, 3, 2468)
REPORT_OBSERVATIONS = (6, 12, 18, 24)
ALL_OBSERVATIONS = tuple(range(1, 25))
BETA_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
DEFAULT_REFERENCE_NPZ = os.path.join(SCRIPT_DIR, 'data', 'reference_forecasts', 'test_references.npz')
CORRECTION_UNIT = 'km east and km north relative to T-CLIPER position'
PERSISTENCE_CACHE = {}
CLIMATOLOGY_CACHE = {}
CLIPER_TRAIN_FOLDS = 5
REFERENCE_METHOD_AUDIT_REQUIRED = None

EMBEDDED_REFERENCE_SOURCE_SHA256 = '52a536dccd613f511625a1fd71e457e5bf57612ce5982dcb7acd00a7470afdcf'

MODEL_CONFIG = {
    'patch_len': 4,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 3,
    'd_ff': 256,
    'dropout': 0.25,
    'kg_embed_dim': 64,
    'fusion_method': 'sdgf_attention',
}

TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0005,
    'weight_decay': 1e-3,
    'early_stopping_patience': 20,
    'scheduler_patience': 7,
    'scheduler_factor': 0.5,
    'grad_clip': 1.0,
    'long_horizon_alpha': 0.0,
    'residual_smooth_lambda': 0.0,
}

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TyphoonRGCNDataset(Dataset):
    def __init__(self, trajectory, correction_km, kg_embeddings, sample_metadata,
                 persistence_coordinates, cliper_coordinates, true_coordinates):
        self.trajectory = torch.from_numpy(trajectory.astype(np.float32))
        self.raw_correction_km = correction_km
        self.target = None

        self.kg_embeddings = torch.from_numpy(kg_embeddings.astype(np.float32))
        self.sample_metadata = sample_metadata
        self.persistence_coordinates = persistence_coordinates
        self.cliper_coordinates = cliper_coordinates
        self.true_coordinates = true_coordinates

    def set_correction_scaler(self, mean_km, std_km):
        if self.raw_correction_km is None:
            return
        normalized = (self.raw_correction_km - mean_km) / std_km
        if not np.isfinite(normalized).all():
            raise ValueError('归一化修正量含非有限值')
        self.target = torch.from_numpy(normalized.astype(np.float32))

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, index):
        if self.target is None:
            return self.trajectory[index], self.kg_embeddings[index]
        return self.trajectory[index], self.kg_embeddings[index], self.target[index]


def correction_from_coordinates(baseline, true):

    if baseline.shape != true.shape or baseline.shape[-1] != 2:
        raise ValueError('T-CLIPER 与真实位置形状必须一致')
    p_lon = np.deg2rad(baseline[..., 0])
    p_lat = np.deg2rad(baseline[..., 1])
    t_lon = np.deg2rad(true[..., 0])
    t_lat = np.deg2rad(true[..., 1])
    east = (EARTH_RADIUS_KM * np.cos(p_lat)
            * persistence_method.wrapped_longitude_difference(t_lon, p_lon))
    north = EARTH_RADIUS_KM * (t_lat - p_lat)
    correction = np.stack((east, north), axis=-1)
    if not np.isfinite(correction).all():
        raise ValueError('真实位置相对 T-CLIPER 的修正量含非有限值')
    return correction


def coordinates_from_correction(baseline, correction_km):

    if baseline.shape != correction_km.shape or baseline.shape[-1] != 2:
        raise ValueError('T-CLIPER 与预测修正量形状必须一致')
    p_lon = np.deg2rad(baseline[..., 0])
    p_lat = np.deg2rad(baseline[..., 1])
    cosine = np.maximum(np.cos(p_lat), 1e-5)
    lon = (np.rad2deg(p_lon + correction_km[..., 0]
                      / (EARTH_RADIUS_KM * cosine)) % 360.0)
    lat = np.rad2deg(p_lat + correction_km[..., 1] / EARTH_RADIUS_KM)
    result = np.stack((lon, lat), axis=-1)
    if not np.isfinite(result).all() or np.any(np.abs(lat) > 90):
        raise ValueError('预测修正量得到无效经纬度')
    return result


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, n_features, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.linear = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        n_patches = seq_len // self.patch_len
        x = x[:, :n_patches * self.patch_len, :]
        x = x.reshape(batch_size, n_patches, self.patch_len * n_features)
        return self.linear(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.ffn(x))


class RGCNTemporalBranch(nn.Module):
    def __init__(self, kg_embed_dim, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.temporal_projection = nn.Sequential(
            nn.Linear(kg_embed_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.patch_projection = nn.Linear(patch_len * d_model, d_model)

    def forward(self, x_rgcn):
        batch_size, seq_len, _ = x_rgcn.shape
        n_patches = seq_len // self.patch_len
        x = self.temporal_projection(x_rgcn)
        x = x[:, :n_patches * self.patch_len, :]
        x = x.reshape(batch_size, n_patches, self.patch_len * x.shape[-1])
        return self.patch_projection(x)


class PatchTSTWithRGCNChannelGated(nn.Module):


    def __init__(self, lookback_window, forecast_horizon,
                 n_trad_features, n_output_features, kg_embed_dim,
                 patch_len, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        if lookback_window % patch_len:
            raise ValueError('历史窗口长度必须能被 patch_len 整除')
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.n_output_features = n_output_features
        self.n_patches = lookback_window // patch_len
        self.patch_embedding = PatchEmbedding(patch_len, n_trad_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.rgcn_branch = RGCNTemporalBranch(
            kg_embed_dim, d_model, patch_len, dropout,
        )
        self.kg_transform = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, d_model), nn.Sigmoid(),
        )
        self.output_proj = nn.Linear(d_model, forecast_horizon * n_output_features)

        del self.gate_net
        self.branch_proj = nn.Linear(self.n_patches * d_model, d_model)
        self.branch_norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(d_model))
        self.key_proj = nn.Linear(d_model, d_model)


    def forward(self, x_trad, x_rgcn, return_gate=False):
        if x_trad.shape[:2] != x_rgcn.shape[:2] or x_trad.shape[1] != self.lookback_window:
            raise ValueError('Trajectory/KG history misalignment')
        temporal = self.dropout(self.patch_embedding(x_trad) + self.pos_encoding)
        for layer in self.encoder_layers:
            temporal = layer(temporal)
        kg = self.kg_transform(self.rgcn_branch(x_rgcn))
        streams = torch.stack((temporal, kg), dim=1)
        features = self.branch_norm(self.branch_proj(streams.flatten(start_dim=2)))
        scores = torch.matmul(self.key_proj(features), self.query)
        weights = scores.softmax(dim=1)
        fused = (streams * weights[:, :, None, None]).sum(dim=1)
        output = self.output_proj(fused.mean(dim=1)).reshape(-1, self.forecast_horizon, self.n_output_features)
        return (output, weights[:, 1:2]) if return_gate else output


def fill_missing_climate_months(velocities, times):

    months = times.astype('datetime64[M]').astype(np.int64) % 12 + 1
    available = set(map(int, np.unique(months)))
    if not available:
        raise ValueError('训练折没有可用于气候场的运动记录')
    extra_velocities = []
    extra_times = []
    fallback = {}
    for month in range(1, 13):
        if month in available:
            continue
        nearest = min(
            available,
            key=lambda candidate: (
                min((candidate - month) % 12, (month - candidate) % 12),
                candidate,
            ),
        )
        selected = velocities[months == nearest]
        extra_velocities.append(selected)
        extra_times.append(np.full(
            len(selected), np.datetime64(f'2001-{month:02d}-15', 'ns'),
            dtype='datetime64[ns]',
        ))
        fallback[str(month)] = nearest
    if fallback:
        velocities = np.concatenate((velocities, *extra_velocities))
        times = np.concatenate((times, *extra_times))
    return velocities, times, fallback


def prepare_climatology(metadata):

    if CLIMATOLOGY_CACHE:
        return CLIMATOLOGY_CACHE
    train = persistence_method.load_split('train', metadata, TRAJECTORY_DIR)
    identifiers = np.asarray([str(value) for value in train['typhoon_ids']])
    folds = np.asarray([
        int.from_bytes(hashlib.sha256(tid.encode('utf-8')).digest()[:4], 'big')
        % CLIPER_TRAIN_FOLDS
        for tid in identifiers
    ], dtype=np.int8)
    velocities, times, _, _ = persistence_method.unique_training_observations(train)
    full_field, _ = persistence_method.build_climatology(velocities, times)
    fold_fields = {}
    for fold in range(CLIPER_TRAIN_FOLDS):
        held_out = folds == fold
        if not held_out.any() or held_out.all():
            raise ValueError(f'CLIPER 训练折 {fold} 为空或包含全部台风')
        subset = {
            key: value[~held_out]
            for key, value in train.items()
        }
        fold_velocities, fold_times, _, _ = (
            persistence_method.unique_training_observations(subset)
        )
        fold_velocities, fold_times, fallback = fill_missing_climate_months(
            fold_velocities, fold_times,
        )
        if fallback:
            print(f'CLIPER 训练折 {fold}: 缺月仅借非留出训练台风的相邻月 '
                  f'{fallback}', flush=True)
        fold_fields[fold], _ = persistence_method.build_climatology(
            fold_velocities, fold_times,
        )
    CLIMATOLOGY_CACHE.update({
        'full_field': full_field,
        'fold_fields': fold_fields,
        'train_fold_by_window': folds,
        'train_typhoon_ids': train['typhoon_ids'].copy(),
        'train_input_time_steps': train['input_time_steps'].copy(),
    })
    return CLIMATOLOGY_CACHE


def load_test_cliper(reference_path, sample_metadata, true_coordinates):

    with np.load(reference_path, allow_pickle=False) as reference:
        required = set(sample_metadata) | {'cliper_predictions', 'ground_truth'}
        missing = required - set(reference.files)
        if missing:
            raise ValueError(f'测试 T-CLIPER 参考文件缺少字段: {sorted(missing)}')
        if any(not np.array_equal(reference[key], value)
               for key, value in sample_metadata.items()):
            raise ValueError('测试 T-CLIPER 与轨迹样本或目标时间未对齐')
        if not np.allclose(reference['ground_truth'], true_coordinates,
                           rtol=0, atol=1e-5):
            raise ValueError('测试 T-CLIPER 的真实轨迹不一致')
        cliper = reference['cliper_predictions'].copy()
    if (cliper.shape != true_coordinates.shape
            or not np.isfinite(cliper).all()):
        raise ValueError('测试 T-CLIPER 轨迹形状或数值无效')
    return cliper


def reference_requires_method_audit(reference_path):

    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(reference_path)),
        'reference_generation_summary.json',
    )
    if not os.path.isfile(summary_path):
        return True
    with open(summary_path, encoding='utf-8') as file:
        summary = json.load(file)
    expected_sources = {
        'train_data_sha256': os.path.join(TRAJECTORY_DIR, 'train_data.npz'),
        'test_data_sha256': os.path.join(TRAJECTORY_DIR, 'test_data.npz'),
        'metadata_sha256': os.path.join(TRAJECTORY_DIR, 'metadata.pkl'),
        'reference_npz_sha256': reference_path,
    }
    for key, path in expected_sources.items():
        if summary.get(key) != file_sha256(path):
            raise ValueError(f'测试 T-CLIPER 参考预报来源已变化: {key}')
    return summary.get('generator_sha256') != EMBEDDED_REFERENCE_SOURCE_SHA256


def load_split(split, kg_embedding_dir, metadata, reference_path):
    trajectory_path = os.path.join(TRAJECTORY_DIR, f'{split}_data.npz')
    kg_path = os.path.join(kg_embedding_dir, f'{split}_causal_embeddings.npz')
    alignment_keys = (
        'typhoon_ids', 'start_indices', 'input_time_steps',
        'forecast_origin_time_steps', 'years',
    )
    sample_keys = (
        'typhoon_ids', 'start_indices', 'input_time_steps',
        'forecast_time_steps', 'forecast_origin_time_steps', 'years',
        'input_times', 'forecast_times', 'forecast_origin_time',
        'forecast_elapsed_hours',
    )
    with np.load(trajectory_path, allow_pickle=False) as trajectory_file, \
            np.load(kg_path, allow_pickle=False) as kg_file:
        for key in ('X', 'y', *sample_keys):
            if key not in trajectory_file:
                raise ValueError(f'{trajectory_path} 缺少 {key}')
        for key in ('embeddings', *alignment_keys):
            if key not in kg_file:
                raise ValueError(f'{kg_path} 缺少 {key}')
        x = trajectory_file['X'].copy()
        y = trajectory_file['y'].copy()
        kg = kg_file['embeddings'].copy()
        sample_metadata = {key: trajectory_file[key].copy() for key in sample_keys}
        for key in alignment_keys:
            if not np.array_equal(trajectory_file[key], kg_file[key]):
                raise ValueError(f'{split}: 轨迹与 KG 的 {key} 不一致')
        time_steps = trajectory_file['input_time_steps']
        origins = trajectory_file['forecast_origin_time_steps']
        if not np.array_equal(time_steps[:, -1], origins):
            raise ValueError(f'{split}: 历史最后一个 OBS 与 forecast origin 不一致')
        if not np.all(np.diff(time_steps, axis=1) > 0):
            raise ValueError(f'{split}: input_time_steps 必须严格递增')
        forecast_steps = trajectory_file['forecast_time_steps']
        if (forecast_steps.shape != (len(x), 24)
                or not np.all(np.diff(forecast_steps, axis=1) > 0)
                or not np.all(forecast_steps[:, 0] > origins)):
            raise ValueError(f'{split}: forecast_time_steps 与预测起点不一致')
        input_times = trajectory_file['input_times']
        forecast_times = trajectory_file['forecast_times']
        origin_times = trajectory_file['forecast_origin_time']
        elapsed_hours = trajectory_file['forecast_elapsed_hours']
        if (input_times.shape != (len(x), 12)
                or forecast_times.shape != (len(x), 24)
                or origin_times.shape != (len(x),)
                or elapsed_hours.shape != (len(x), 24)
                or not np.issubdtype(input_times.dtype, np.datetime64)
                or not np.issubdtype(forecast_times.dtype, np.datetime64)
                or not np.issubdtype(origin_times.dtype, np.datetime64)
                or np.isnat(input_times).any()
                or np.isnat(forecast_times).any()
                or np.isnat(origin_times).any()
                or not np.array_equal(input_times[:, -1], origin_times)
                or not np.all(np.diff(input_times, axis=1) > np.timedelta64(0, 'ns'))
                or not np.all(np.diff(forecast_times, axis=1) > np.timedelta64(0, 'ns'))
                or not np.all(forecast_times[:, 0] > origin_times)
                or not np.allclose(
                    elapsed_hours,
                    (forecast_times - origin_times[:, None]) / np.timedelta64(1, 'h'),
                )):
            raise ValueError(f'{split}: 原始时间戳或实际预测提前小时数不一致')

    n = len(x)
    if x.shape != (n, 12, 4) or y.shape != (n, 24, 2):
        raise ValueError(f'{split}: 期望轨迹 (N,12,4)、目标 (N,24,2)，得到 {x.shape}、{y.shape}')
    if kg.shape != (n, 12, MODEL_CONFIG['kg_embed_dim']):
        raise ValueError(f'{split}: 期望 KG (N,12,64)，得到 {kg.shape}')
    if any(len(values) != n for values in sample_metadata.values()):
        raise ValueError(f'{split}: 测试样本元数据与预测数组行数不一致')
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(kg).all()):
        raise ValueError(f'{split}: 输入或目标含非有限值')
    cached = PERSISTENCE_CACHE.get(split)
    if cached is not None:
        if (not np.array_equal(x, cached['x'])
                or not np.array_equal(y, cached['y'])
                or any(not np.array_equal(sample_metadata[key], cached[key])
                       for key in sample_keys)):
            raise ValueError(f'{split}: 当前数据与缓存的历史 Persistence 输入不一致')
        persistence = cached['persistence']
        cliper = cached['cliper']
        true_coordinates = cached['true_coordinates']
        correction_km = cached['correction_km']
    else:
        input_coordinates = metadata['scaler_X'].inverse_transform(
            x.reshape(-1, 4)
        ).reshape(n, 12, 4)[..., :2]
        true_coordinates = metadata['scaler_y'].inverse_transform(
            y.reshape(-1, 2)
        ).reshape(n, 24, 2)
        persistence = np.empty((n, 24, 2), dtype=np.float64)
        if split == 'test':
            cliper = load_test_cliper(
                reference_path, sample_metadata, true_coordinates,
            )
            climate = None
            audit_field = (
                prepare_climatology(metadata)['full_field']
                if REFERENCE_METHOD_AUDIT_REQUIRED else None
            )
        else:
            climate = prepare_climatology(metadata)
            audit_field = None
            cliper = np.empty((n, 24, 2), dtype=np.float64)
            if split == 'train' and (
                    not np.array_equal(sample_metadata['typhoon_ids'],
                                       climate['train_typhoon_ids'])
                    or not np.array_equal(sample_metadata['input_time_steps'],
                                          climate['train_input_time_steps'])):
                raise ValueError('训练窗口次序与按台风分折的气候场不一致')
        for index in range(n):
            motion, _, _ = persistence_method.initial_motion(
                input_coordinates[index], sample_metadata['input_times'][index],
                sample_metadata['input_time_steps'][index],
            )
            persistence[index] = persistence_method.integrate(
                input_coordinates[index, -1], motion,
                sample_metadata['forecast_origin_time'][index],
                sample_metadata['forecast_elapsed_hours'][index], fields=None,
            )
            if audit_field is not None:
                audited = persistence_method.integrate(
                    input_coordinates[index, -1], motion,
                    sample_metadata['forecast_origin_time'][index],
                    sample_metadata['forecast_elapsed_hours'][index],
                    fields=audit_field,
                )
                if not np.allclose(cliper[index], audited, rtol=0, atol=1e-6):
                    raise ValueError(
                        f'测试窗口 {index} 的已保存 T-CLIPER 与当前训练期气候场不一致'
                    )
            if split != 'test':
                fields = (climate['fold_fields'][
                    int(climate['train_fold_by_window'][index])
                ] if split == 'train' else climate['full_field'])
                cliper[index] = persistence_method.integrate(
                    input_coordinates[index, -1], motion,
                    sample_metadata['forecast_origin_time'][index],
                    sample_metadata['forecast_elapsed_hours'][index],
                    fields=fields,
                )
        correction_km = (
            correction_from_coordinates(cliper, true_coordinates)
            if split != 'test' else None
        )
        PERSISTENCE_CACHE[split] = {
            'x': x, 'y': y, **sample_metadata,
            'persistence': persistence,
            'cliper': cliper,
            'true_coordinates': true_coordinates,
            'correction_km': correction_km,
        }
    print(f'{split}: 轨迹 {x.shape}，KG {kg.shape}，目标 {y.shape}；'
          '历史 Persistence、T-CLIPER 与逐样本对齐通过')
    return TyphoonRGCNDataset(
        x, correction_km, kg, sample_metadata,
        persistence, cliper, true_coordinates,
    )


def load_data(kg_embedding_dir, expected_seed, reference_path):
    global REFERENCE_METHOD_AUDIT_REQUIRED
    if REFERENCE_METHOD_AUDIT_REQUIRED is None:
        REFERENCE_METHOD_AUDIT_REQUIRED = reference_requires_method_audit(
            reference_path,
        )
    with open(os.path.join(kg_embedding_dir, 'causal_embedding_summary.json'),
              encoding='utf-8') as file:
        kg_summary = json.load(file)
    rgcn_dir = os.path.dirname(os.path.abspath(kg_summary['model_checkpoint']))
    expected_checkpoint = os.path.normcase(os.path.abspath(
        os.path.join(rgcn_dir, 'rgcn_train_only.pth')
    ))
    actual_checkpoint = os.path.normcase(os.path.abspath(
        kg_summary['model_checkpoint']
    ))
    if kg_summary['seed'] != expected_seed or actual_checkpoint != expected_checkpoint:
        raise ValueError(f'seed={expected_seed}: 因果嵌入与本次 R-GCN 检查点不匹配')
    with open(os.path.join(rgcn_dir, 'rgcn_config.json'), encoding='utf-8') as file:
        rgcn_config = json.load(file)
    if rgcn_config['seed'] != expected_seed:
        raise ValueError(f'seed={expected_seed}: R-GCN 配置中的 seed 不匹配')
    if rgcn_config['negative_sampling'] != 'filtered_type_constrained_train_nodes':
        raise ValueError(f'seed={expected_seed}: R-GCN 未使用 filtered typed negative sampling')
    with open(os.path.join(TRAJECTORY_DIR, 'metadata.pkl'), 'rb') as file:
        metadata = pickle.load(file)
    if (metadata['input_features'] != ['lng', 'lat', 'speed', 'pressure']
            or metadata['target_features'] != ['lng', 'lat']
            or metadata['lookback_window'] != 12
            or metadata['forecast_horizon'] != 24
            or metadata['n_input_features'] != 4
            or metadata['n_target_features'] != 2):
        raise ValueError('特征工程元数据与 12×4 → 24×2 任务不一致')
    if (metadata['scaler_X'].n_features_in_ != 4
            or metadata['scaler_y'].n_features_in_ != 2):
        raise ValueError('输入与目标 scaler 维度必须分别为 4 和 2')
    datasets = {
        split: load_split(split, kg_embedding_dir, metadata, reference_path)
        for split in ('train', 'val', 'test')
    }
    mean_km = datasets['train'].raw_correction_km.reshape(-1, 2).mean(axis=0)
    std_km = datasets['train'].raw_correction_km.reshape(-1, 2).std(axis=0)
    if (mean_km.shape != (2,) or std_km.shape != (2,)
            or not np.isfinite(mean_km).all()
            or not np.isfinite(std_km).all()
            or not np.all(std_km > 0)):
        raise ValueError('训练集修正量归一化参数无效')
    for dataset in datasets.values():
        dataset.set_correction_scaler(mean_km, std_km)
    metadata['correction_mean_km'] = mean_km
    metadata['correction_std_km'] = std_km
    for split, dataset in datasets.items():
        if kg_summary['splits'][split]['samples'] != len(dataset):
            raise ValueError(f'seed={expected_seed}: {split} 嵌入摘要样本数不一致')
    return datasets, metadata


def make_loaders(datasets, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            datasets[split], batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=(split == 'train'), num_workers=0,
            generator=generator if split == 'train' else None,
        )
        for split in ('train', 'val', 'test')
    }


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for trajectory, kg, target in loader:
        trajectory, kg, target = (
            trajectory.to(DEVICE), kg.to(DEVICE), target.to(DEVICE)
        )
        optimizer.zero_grad()
        loss = criterion(model(trajectory, kg), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['grad_clip'])
        optimizer.step()
        total += loss.item() * len(trajectory)
    return total / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for trajectory, kg, target in loader:
            trajectory, kg, target = (
                trajectory.to(DEVICE), kg.to(DEVICE), target.to(DEVICE)
            )
            total += criterion(model(trajectory, kg), target).item() * len(trajectory)
    return total / len(loader.dataset)


def validation_ade(model, loader, metadata, beta=1.0):

    model.eval()
    outputs = []
    with torch.no_grad():
        for trajectory, kg, _ in loader:
            output = model(trajectory.to(DEVICE), kg.to(DEVICE))
            outputs.append(output.cpu().numpy())
    correction_norm = np.concatenate(outputs)
    correction_km = (
        correction_norm.astype(np.float64) * metadata['correction_std_km']
        + metadata['correction_mean_km']
    )
    prediction = (loader.dataset.cliper_coordinates
                  if beta == 0 else coordinates_from_correction(
                      loader.dataset.cliper_coordinates, beta * correction_km,
                  ))
    ade = float(np.mean(haversine_distance_km(
        prediction, loader.dataset.true_coordinates,
    )))
    if not np.isfinite(ade):
        raise ValueError('验证集 ADE 非有限数')
    return ade


def haversine_distance_km(pred, true):

    pred_lon, pred_lat = np.radians(pred[..., 0]), np.radians(pred[..., 1])
    true_lon, true_lat = np.radians(true[..., 0]), np.radians(true[..., 1])
    delta_lon = pred_lon - true_lon
    delta_lat = pred_lat - true_lat
    a = (np.sin(delta_lat / 2) ** 2
         + np.cos(true_lat) * np.cos(pred_lat) * np.sin(delta_lon / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_reference_distances(reference_path, sample_metadata, true):


    if not os.path.isfile(reference_path):
        raise FileNotFoundError(
            f'缺少 Persistence/CLIPER 参考预报: {reference_path}；'
            '请先按 C5 定义建立同序 test_references.npz'
        )
    alignment_keys = (
        'typhoon_ids', 'start_indices', 'input_time_steps',
        'forecast_time_steps', 'forecast_origin_time_steps', 'years',
        'input_times', 'forecast_times', 'forecast_origin_time',
        'forecast_elapsed_hours',
    )
    required = set(alignment_keys) | {
        'ground_truth', 'persistence_predictions', 'cliper_predictions',
        'persistence_definition', 'cliper_definition',
    }
    with np.load(reference_path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f'参考预报缺少字段: {sorted(missing)}')
        for key in alignment_keys:
            if not np.array_equal(data[key], sample_metadata[key]):
                raise ValueError(f'参考预报的 {key} 与测试样本不一致')
        if (data['ground_truth'].shape != true.shape
                or not np.allclose(data['ground_truth'], true, rtol=0, atol=1e-5)):
            raise ValueError('参考预报的真实轨迹与模型测试集不一致')
        definitions = {}
        distances = {}
        for reference in ('persistence', 'cliper'):
            definition = data[f'{reference}_definition']
            if definition.shape != () or not str(definition.item()).strip():
                raise ValueError(f'{reference} 缺少可审计的参考预报定义')
            definitions[reference] = str(definition.item())
            if (reference == 'cliper'
                    and not all(term in definitions[reference]
                                for term in ('J-North', 'Barnes', 'Tu=70 h',
                                             'Tv=46 h', 'training-only'))):
                raise ValueError('CLIPER 参考预报必须是训练期气候场构建的 J-North T-CLIPER 轨迹积分')
            prediction = data[f'{reference}_predictions']
            if prediction.shape != true.shape or not np.isfinite(prediction).all():
                raise ValueError(f'{reference} 预测形状或数值异常')
            distances[reference] = haversine_distance_km(prediction, true)
    return distances, definitions


def calculate_metrics(pred, true, distance, origin_coordinates,
                      reference_distances=None):
    errors = pred - true
    if pred.shape != true.shape or pred.shape[1:] != (24, 2):
        raise ValueError('评价需要相同形状的 (N,24,2) 经度纬度数组')
    if distance.shape != pred.shape[:2]:
        raise ValueError('距离误差必须具有 (N,24) 形状')
    if origin_coordinates.shape != (len(pred), 2):
        raise ValueError('forecast origin 经纬度必须具有 (N,2) 形状')
    if (not np.isfinite(pred).all() or not np.isfinite(true).all()
            or not np.isfinite(distance).all()
            or not np.isfinite(origin_coordinates).all()):
        raise ValueError('评价数组含非有限值')
    if reference_distances is not None:
        for reference in ('persistence', 'cliper'):
            if reference not in reference_distances:
                raise ValueError(f'缺少 {reference} 参考预报')
            ref = reference_distances[reference]
            if ref.shape != distance.shape or not np.isfinite(ref).all():
                raise ValueError(f'{reference} 参考距离形状或数值异常')
            if np.mean(ref) <= 0 or np.any(np.mean(ref, axis=0) <= 0):
                raise ValueError(f'{reference} 参考误差为零，技巧得分无定义')
    result = {
        'lon_mae_deg': float(np.mean(np.abs(errors[..., 0]))),
        'lon_rmse_deg': float(np.sqrt(np.mean(errors[..., 0] ** 2))),
        'lat_mae_deg': float(np.mean(np.abs(errors[..., 1]))),
        'lat_rmse_deg': float(np.sqrt(np.mean(errors[..., 1] ** 2))),
        'ade_km': float(np.mean(distance)),
        'fde_km': float(np.mean(distance[:, -1])),
        'rmse_de_km': float(np.sqrt(np.mean(distance ** 2))),
        'median_de_km': float(np.median(distance)),
        'p90_de_km': float(np.percentile(distance, 90)),
        'p95_de_km': float(np.percentile(distance, 95)),
    }
    if reference_distances is not None:
        for reference, ref in reference_distances.items():
            result[f'ade_skill_vs_{reference}_pct'] = float(
                (1 - np.mean(distance) / np.mean(ref)) * 100
            )
    for observation in ALL_OBSERVATIONS:
        observation_error = distance[:, observation - 1]
        result[f'de_obs_{observation}_mean_km'] = float(np.mean(observation_error))
        result[f'de_obs_{observation}_rmse_km'] = float(
            np.sqrt(np.mean(observation_error ** 2))
        )
        for coordinate, index in (('lon', 0), ('lat', 1)):
            coordinate_error = errors[:, observation - 1, index]
            result[f'{coordinate}_mae_obs_{observation}_deg'] = float(
                np.mean(np.abs(coordinate_error))
            )
            result[f'{coordinate}_rmse_obs_{observation}_deg'] = float(
                np.sqrt(np.mean(coordinate_error ** 2))
            )
        if reference_distances is not None:
            for reference, ref in reference_distances.items():
                reference_mean = float(np.mean(ref[:, observation - 1]))
                result[f'{reference}_de_obs_{observation}_mean_km'] = reference_mean
                result[f'skill_vs_{reference}_obs_{observation}_pct'] = float(
                    (1 - np.mean(observation_error) / reference_mean) * 100
                )
    return result


def test_model(model, loader, metadata, seed_dir, reference_path, beta):

    model.eval()
    predictions, gates, origins = [], [], []
    with torch.no_grad():
        for trajectory, kg in loader:
            prediction, gate = model(
                trajectory.to(DEVICE), kg.to(DEVICE), return_gate=True,
            )
            predictions.append(prediction.cpu().numpy())
            gates.append(gate.cpu().numpy())
            origins.append(trajectory[:, -1, :].numpy())

    correction_norm = np.concatenate(predictions)
    gates = np.concatenate(gates)
    origin_norm = np.concatenate(origins)
    correction_km = (
        correction_norm.astype(np.float64) * metadata['correction_std_km']
        + metadata['correction_mean_km']
    )
    persistence = loader.dataset.persistence_coordinates
    cliper = loader.dataset.cliper_coordinates
    pred = (cliper.copy() if beta == 0
            else coordinates_from_correction(cliper, beta * correction_km))
    true = loader.dataset.true_coordinates
    origin_coordinates = metadata['scaler_X'].inverse_transform(origin_norm)[:, :2]
    distance = haversine_distance_km(pred, true)
    if not (np.isfinite(pred).all() and np.isfinite(true).all()
            and np.isfinite(distance).all()
            and np.isfinite(origin_coordinates).all()):
        raise ValueError('测试预测或距离误差含非有限值')
    if len(loader.dataset.sample_metadata['typhoon_ids']) != len(pred):
        raise ValueError('测试预测行数与样本索引行数不一致')
    reference_distances, _ = load_reference_distances(
        reference_path, loader.dataset.sample_metadata, true,
    )
    with np.load(reference_path, allow_pickle=False) as reference:
        if not np.allclose(
                persistence, reference['persistence_predictions'],
                rtol=0, atol=1e-6):
            raise ValueError('测试集历史生成的 Persistence 与参考预报不一致')
        if not np.array_equal(cliper, reference['cliper_predictions']):
            raise ValueError('测试集 T-CLIPER 与参考预报不一致')
    metrics = calculate_metrics(
        pred, true, distance, origin_coordinates, reference_distances,
    )

    np.save(os.path.join(seed_dir, 'predictions.npy'), pred)
    np.save(os.path.join(seed_dir, 'ground_truth.npy'), true)
    np.save(os.path.join(seed_dir, 'distance_errors.npy'), distance)
    np.save(os.path.join(seed_dir, 'persistence_predictions.npy'), persistence)
    np.save(os.path.join(seed_dir, 'cliper_predictions.npy'), cliper)

    np.save(os.path.join(seed_dir, 'correction_predictions_km.npy'), correction_km)
    np.save(os.path.join(seed_dir, 'block_channel_gates.npy'), gates)
    np.save(os.path.join(seed_dir, 'forecast_origin_coordinates.npy'), origin_coordinates)
    np.savez_compressed(
        os.path.join(seed_dir, 'test_sample_metadata.npz'),
        **loader.dataset.sample_metadata,
    )
    return metrics


import csv
import shutil
from pathlib import Path

METHOD_ID='motion_conditioned_kg'
METHOD_TAG='TRGF-KG3C1-MotionKG'
def same_path(left, right):
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right))


def ensure_seed_embeddings(seed, causal_root, rgcn_root, kg_dir):

    rgcn_dir = os.path.join(rgcn_root, f'seed_{seed}')
    causal_dir = os.path.join(causal_root, f'seed_{seed}')
    checkpoint = os.path.join(rgcn_dir, 'rgcn_train_only.pth')
    if not os.path.isdir(rgcn_dir):
        raise FileNotFoundError(f'Missing precomputed RGCN: {rgcn_dir}')
    required_rgcn = (
        'rgcn_train_only.pth', 'rgcn_config.json',
        'kg_normalization_stats.npz', 'train_node_global_ids.npy',
        'training_summary.json',
    )
    missing = [name for name in required_rgcn if not os.path.isfile(
        os.path.join(rgcn_dir, name))]
    if missing:
        raise ValueError(f'seed={seed}: RGCN 目录不完整，缺少 {missing}: {rgcn_dir}')
    with open(os.path.join(rgcn_dir, 'rgcn_config.json'), encoding='utf-8') as file:
        rgcn_config = json.load(file)
    if (rgcn_config['seed'] != seed
            or not same_path(rgcn_config['kg_input_dir'], kg_dir)
            or rgcn_config['num_orig_relations'] != 14
            or rgcn_config['num_relations'] != 28
            or rgcn_config['negative_sampling'] !=
            'filtered_type_constrained_train_nodes'):
        raise ValueError(f'seed={seed}: RGCN 检查点配置与当前派生 KG 不一致')

    if not os.path.isdir(causal_dir):
        raise FileNotFoundError(f'Missing precomputed causal KG: {causal_dir}')
    summary_path = os.path.join(causal_dir, 'causal_embedding_summary.json')
    required_causal = [
        summary_path,
        os.path.join(causal_dir, 'causal_sensitivity_check.json'),
        *(os.path.join(causal_dir, f'{split}_causal_embeddings.npz')
          for split in ('train', 'val', 'test')),
    ]
    missing = [path for path in required_causal if not os.path.isfile(path)]
    if missing:
        raise ValueError(f'seed={seed}: 因果嵌入目录不完整，缺少 {missing}')
    with open(summary_path, encoding='utf-8') as file:
        summary = json.load(file)
    if (summary['seed'] != seed
            or not same_path(summary['model_checkpoint'], checkpoint)
            or not same_path(summary['kg_dir'], kg_dir)
            or not same_path(summary['processed_dir'], TRAJECTORY_DIR)
            or summary['causal_sensitivity_passed'] is not True
            or any(os.path.getmtime(path) < os.path.getmtime(checkpoint)
                   for path in required_causal)):
        raise ValueError(f'seed={seed}: 因果嵌入与当前 RGCN 检查点不一致')
    print(f'seed={seed}: RGCN 与因果嵌入已就绪', flush=True)
    return causal_dir


def train_sdgf_validation(seed, datasets, metadata, seed_dir, reference_path):
    set_seed(seed)
    loaders = make_loaders(datasets, seed)
    model = PatchTSTWithRGCNChannelGated(
        lookback_window=metadata['lookback_window'],
        forecast_horizon=metadata['forecast_horizon'],
        n_trad_features=metadata['n_input_features'],
        n_output_features=metadata['n_target_features'],
        kg_embed_dim=MODEL_CONFIG['kg_embed_dim'],
        patch_len=MODEL_CONFIG['patch_len'],
        d_model=MODEL_CONFIG['d_model'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_layers=MODEL_CONFIG['n_layers'],
        d_ff=MODEL_CONFIG['d_ff'],
        dropout=MODEL_CONFIG['dropout'],
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=TRAIN_CONFIG['scheduler_factor'],
        patience=TRAIN_CONFIG['scheduler_patience'],
    )
    checkpoint_path = os.path.join(seed_dir, 'best_model.pth')
    best_val_loss = float('inf')
    val_loss_at_best_ade = float('inf')
    best_val_ade = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
        train_loss = train_epoch(model, loaders['train'], criterion, optimizer)
        val_loss = validate(model, loaders['val'], criterion)
        val_ade = validation_ade(model, loaders['val'], metadata)
        best_val_loss = min(best_val_loss, val_loss)
        scheduler.step(val_ade)
        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_ade_km': val_ade,
        })
        print(f'seed={seed} epoch={epoch:3d} train={train_loss:.6f} '
              f'val={val_loss:.6f} val_ADE={val_ade:.3f} km')
        if val_ade < best_val_ade:
            val_loss_at_best_ade = val_loss
            best_val_ade = val_ade
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'seed': seed, 'epoch': epoch, 'val_loss': val_loss,
                'val_ade_km': val_ade,
                'model_state_dict': model.state_dict(),
                'model_config': MODEL_CONFIG,
                'long_horizon_alpha': TRAIN_CONFIG['long_horizon_alpha'],
                'residual_smooth_lambda': TRAIN_CONFIG['residual_smooth_lambda'],
                'correction_mean_km': metadata['correction_mean_km'].tolist(),
                'correction_std_km': metadata['correction_std_km'].tolist(),
            }, checkpoint_path)
        else:
            patience_counter += 1
        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            break

    with open(os.path.join(seed_dir, 'training_history.json'), 'w', encoding='utf-8') as file:
        json.dump(history, file, indent=2)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    beta_ade_km = {
        beta: validation_ade(model, loaders['val'], metadata, beta)
        for beta in BETA_CANDIDATES
    }
    best_beta = min(BETA_CANDIDATES, key=beta_ade_km.__getitem__)
    print(f'seed={seed} validation beta={best_beta:g} '
          f'ADE={beta_ade_km[best_beta]:.3f} km')
    checkpoint['best_beta'] = best_beta
    checkpoint['run_lock_sha256'] = RUN_FINGERPRINT
    torch.save(checkpoint, checkpoint_path)
    record = {
        'seed': seed, 'method': 'sdgf_attention', 'best_epoch': best_epoch,
        'best_beta': best_beta, 'best_val_loss': best_val_loss,
        'val_loss_at_best_ade': val_loss_at_best_ade,
        'best_val_ade_beta1_km': best_val_ade,
        'selected_val_ade_km': beta_ade_km[best_beta],
        'validation_beta_ade_km': {str(k): v for k, v in beta_ade_km.items()},
        'epochs_completed': len(history), 'training_seconds': time.time()-start_time,
        'correction_mean_km': metadata['correction_mean_km'].tolist(),
        'correction_std_km': metadata['correction_std_km'].tolist(),
        'model_config': MODEL_CONFIG, 'train_config': TRAIN_CONFIG,
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'run_lock_sha256': RUN_FINGERPRINT, 'test_used_for_selection': False,
    }
    write_json(Path(seed_dir)/'validation.json', record)
    return record


REFINEMENT_CONFIG = {
    'epochs':80, 'learning_rate':.001, 'batch_size':32, 'patience':15,
    'weight_decay':0., 'forecast_delta_penalty':.1,
    'scheduler_patience':5, 'scheduler_factor':.5, 'grad_clip':1.,
    'modulation_bound':.1,
    'criterion':'beta-calibrated normalized forecast MSE + .1 deviation from frozen SDGF',
    'selection':'fixed-beta validation ADE, including identity epoch0; retain all seeds',
}

class MotionConditionedKG(PatchTSTWithRGCNChannelGated):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d=self.output_proj.in_features
        rng=torch.random.get_rng_state()
        self.context_norm=nn.LayerNorm(d)
        self.conditioner=nn.Sequential(nn.Linear(2*d,32), nn.GELU(), nn.Linear(32,2*d))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        torch.random.set_rng_state(rng)

    @torch.no_grad()
    def encode(self, x, k):
        if x.shape[:2]!=k.shape[:2] or x.shape[1]!=self.lookback_window:
            raise ValueError('Trajectory/KG history alignment failed')
        t=self.dropout(self.patch_embedding(x)+self.pos_encoding)
        for layer in self.encoder_layers: t=layer(t)
        kg=self.kg_transform(self.rgcn_branch(k))
        streams=torch.stack((t,kg),1)
        weights=(self.key_proj(self.branch_norm(self.branch_proj(
            streams.flatten(start_dim=2))))@self.query).softmax(1)
        history=x.clone()
        history[:,:,:2]=x[:,:,:2]-x[:,:1,:2]
        return t,kg,weights,self.patch_embedding(history)

    def modulation(self, motion):
        c=self.context_norm(motion)
        scale,shift=self.conditioner(torch.cat((c,c-c[:,:1,:]),-1)).chunk(2,-1)
        limit=REFINEMENT_CONFIG['modulation_bound']
        return 1+limit*torch.tanh(scale),limit*torch.tanh(shift)

    def predict_features(self, t, kg, w, motion):
        gamma,bias=self.modulation(motion)
        adjusted=gamma*kg+bias
        fused=w[:,0,None,None]*t+w[:,1,None,None]*adjusted
        plain=w[:,0,None,None]*t+w[:,1,None,None]*kg
        shape=(-1,self.forecast_horizon,self.n_output_features)
        return (self.output_proj(fused.mean(1)).reshape(shape),
                self.output_proj(plain.mean(1)).reshape(shape))

    def forward(self, x, k, return_gate=False):
        features=self.encode(x,k)
        prediction,_=self.predict_features(*features)
        return (prediction,features[2]) if return_gate else prediction

def make_motion_model(seed, metadata, base_folder):
    set_seed(seed)
    model=MotionConditionedKG(**model_kwargs(metadata)).to(DEVICE).eval()
    checkpoint=torch.load(Path(base_folder)/'best_model.pth', map_location=DEVICE, weights_only=True)
    state=checkpoint['model_state_dict']
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys or not all(n.startswith(('context_norm.','conditioner.'))
                                       for n in result.missing_keys):
        raise ValueError('Incompatible SDGF checkpoint')
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('context_norm.','conditioner.')))
    for name,value in state.items():
        if not torch.equal(value,model.state_dict()[name]):
            raise ValueError('Original parameter changed: '+name)
    return model,state

def cache_features(model, loader, seed, shuffle):
    parts=[[],[],[],[],[]]
    for x,k,y in DataLoader(loader.dataset,batch_size=128,shuffle=False,num_workers=0):
        features=model.encode(x.to(DEVICE),k.to(DEVICE))
        for dest,value in zip(parts[:4],features):dest.append(value.cpu())
        parts[4].append(y)
    tensors=[torch.cat(v) for v in parts]
    return DataLoader(TensorDataset(*tensors),batch_size=REFINEMENT_CONFIG['batch_size'],
                      shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=0)

def cached_validation_metrics(model, cached, source, metadata, beta):
    model.eval();pred=[]
    with torch.no_grad():
        for t,k,w,c,_ in cached:
            pred.append(model.predict_features(*(v.to(DEVICE) for v in (t,k,w,c)))[0].cpu().numpy())
    return position_metrics_from_normalized(np.concatenate(pred),source,metadata,beta)

def train_motion_validation(seed, datasets, metadata, seed_dir, base_dir):
    folder=Path(seed_dir);base=read_json(Path(base_dir)/'validation.json')
    model,frozen=make_motion_model(seed,metadata,base_dir)
    loaders=make_loaders(datasets,seed);beta=base['best_beta']
    train_cache=cache_features(model,loaders['train'],seed,True)
    val_cache=cache_features(model,loaders['val'],seed,False)
    xx,kk,_=next(iter(loaders['val']))
    with torch.no_grad():
        features=model.encode(xx.to(DEVICE),kk.to(DEVICE))
        torch.testing.assert_close(model.predict_features(*features)[0],
                                   model(xx.to(DEVICE),kk.to(DEVICE)),rtol=0,atol=0)
    best_metrics=cached_validation_metrics(model,val_cache,loaders['val'],metadata,beta)
    if abs(best_metrics['ade_km']-base['selected_val_ade_km'])>1e-4:
        raise ValueError('Identity initialization differs from SDGF')
    checkpoint=folder/'best_model.pth';best=best_metrics['ade_km']
    best_epoch=0;stale=0;history=[];started=time.time()
    def save(epoch):
        torch.save({'seed':seed,'epoch':epoch,'best_beta':beta,
            'method':'motion_conditioned_kg','model_state_dict':model.state_dict(),
            'model_config':MODEL_CONFIG,'refinement_config':REFINEMENT_CONFIG,
            'correction_mean_km':metadata['correction_mean_km'].tolist(),
            'correction_std_km':metadata['correction_std_km'].tolist(),
            'base_checkpoint_sha256':file_sha256(Path(base_dir)/'best_model.pth'),
            'run_lock_sha256':RUN_FINGERPRINT},checkpoint)
    save(0)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=optim.Adam(parameters,lr=REFINEMENT_CONFIG['learning_rate'],weight_decay=0.)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
    offset=torch.as_tensor((beta-1)*metadata['correction_mean_km']/metadata['correction_std_km'],
                           dtype=torch.float32,device=DEVICE)
    for epoch in range(1,REFINEMENT_CONFIG['epochs']+1):
        model.eval();total=0.
        for t,k,w,c,y in train_cache:
            t,k,w,c,y=(v.to(DEVICE) for v in (t,k,w,c,y))
            optimizer.zero_grad();pred,plain=model.predict_features(t,k,w,c)
            loss=(beta*pred-y+offset).square().mean()+.1*(pred-plain).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite refinement loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            total+=loss.item()*len(y)
        values=cached_validation_metrics(model,val_cache,loaders['val'],metadata,beta)
        ade=values['ade_km'];scheduler.step(ade)
        history.append({'epoch':epoch,'train_loss':total/len(train_cache.dataset),
                        'validation_metrics':values})
        if epoch==1 or epoch%5==0:
            print(f'MotionKG seed={seed} epoch={epoch} beta={beta} val_ADE={ade:.6f}',flush=True)
        if ade<best:best=ade;best_metrics=values;best_epoch=epoch;stale=0;save(epoch)
        else:stale+=1
        if stale>=REFINEMENT_CONFIG['patience']:break
    model.load_state_dict(torch.load(checkpoint,map_location=DEVICE,weights_only=True)['model_state_dict'])
    for name,value in frozen.items():
        if not torch.equal(value,model.state_dict()[name]):raise ValueError('Frozen parameter changed: '+name)
    live=validation_metrics(model,loaders['val'],metadata,beta)
    if abs(live['ade_km']-best)>1e-4:raise ValueError('Cached/live validation mismatch')
    record={'seed':seed,'method':'motion_conditioned_kg','best_epoch':best_epoch,
        'best_beta':beta,'selected_val_ade_km':live['ade_km'],'validation_metrics':live,
        'baseline_val_ade_km':base['selected_val_ade_km'],
        'epochs_completed':len(history),'training_seconds':time.time()-started,
        'num_trainable_parameters':sum(p.numel() for p in parameters),
        'original_parameters_unchanged':True,'test_used_for_selection':False,
        'correction_mean_km':metadata['correction_mean_km'].tolist(),
        'correction_std_km':metadata['correction_std_km'].tolist(),
        'base_checkpoint_sha256':file_sha256(Path(base_dir)/'best_model.pth'),
        'checkpoint_sha256':file_sha256(checkpoint),'run_lock_sha256':RUN_FINGERPRINT}
    write_json(folder/'training_history.json',history);write_json(folder/'validation.json',record)
    del model,train_cache,val_cache
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return record

RUN_FINGERPRINT=None
DEFAULT_UPSTREAM=os.path.join(SCRIPT_DIR,'perform','result','TRGF-KG3C1-nine-methods_20261003')
OVERVIEW_METRICS=('lon_mae_deg','lon_rmse_deg','lat_mae_deg','lat_rmse_deg',
                  'ade_km','fde_km','rmse_de_km','median_de_km','p90_de_km','p95_de_km',
                  'ade_skill_vs_persistence_pct','ade_skill_vs_cliper_pct')

def read_json(path):
    with open(path,encoding='utf-8') as f:return json.load(f)

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def model_kwargs(metadata):
    return dict(lookback_window=metadata['lookback_window'],forecast_horizon=metadata['forecast_horizon'],
        n_trad_features=metadata['n_input_features'],n_output_features=metadata['n_target_features'],
        **{k:MODEL_CONFIG[k] for k in ('kg_embed_dim','patch_len','d_model','n_heads','n_layers','d_ff','dropout')})

def position_metrics_from_normalized(prediction,source,metadata,beta):
    correction=prediction.astype(np.float64)*metadata['correction_std_km']+metadata['correction_mean_km']
    positions=(source.dataset.cliper_coordinates if beta==0 else
               coordinates_from_correction(source.dataset.cliper_coordinates,beta*correction))
    truth=source.dataset.true_coordinates;error=positions-truth
    return {'lon_mae_deg':float(np.mean(abs(error[...,0]))),
        'lon_rmse_deg':float(np.sqrt(np.mean(error[...,0]**2))),
        'lat_mae_deg':float(np.mean(abs(error[...,1]))),
        'lat_rmse_deg':float(np.sqrt(np.mean(error[...,1]**2))),
        'ade_km':float(np.mean(haversine_distance_km(positions,truth)))}

def validation_metrics(model,loader,metadata,beta):
    model.eval();pred=[]
    with torch.no_grad():
        for x,k,_ in loader:pred.append(model(x.to(DEVICE),k.to(DEVICE)).cpu().numpy())
    return position_metrics_from_normalized(np.concatenate(pred),loader,metadata,beta)

def prepare_run(args):
    global RUN_FINGERPRINT
    if len(set(args.seeds))!=len(args.seeds) or any(s not in SEEDS for s in args.seeds):
        raise ValueError(f'Choose non-repeating seeds from {SEEDS}')
    args.output_dir=os.path.abspath(args.output_dir or os.path.join(RESULT_DIR,METHOD_TAG+'_'+
        datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')))
    output=Path(args.output_dir)
    if output.exists() and args.phase!='evaluate-test' and not args.resume:
        raise FileExistsError('Output exists; use --resume --output-dir or choose a new directory')
    if args.phase=='evaluate-test' and not output.exists():raise FileNotFoundError(output)
    previous=read_json(output/'run_config.json') if (output/'run_config.json').exists() else None
    if METHOD_ID=='motion_conditioned_kg' and not args.sdgf_run and previous:
        args.sdgf_run=previous.get('sdgf_run')
    for s in args.seeds:ensure_seed_embeddings(s,args.causal_root,args.rgcn_root,args.kg_dir)
    inputs=[Path(TRAJECTORY_DIR)/'metadata.pkl',Path(args.reference_path),
            Path(args.kg_dir)/'export_config.json',
            *(Path(TRAJECTORY_DIR)/(s+'_data.npz') for s in ('train','val','test'))]
    for seed in args.seeds:
        inputs.extend(sorted(p for p in (Path(args.causal_root)/f'seed_{seed}').iterdir() if p.is_file()))
        inputs.extend(Path(args.rgcn_root)/f'seed_{seed}'/name for name in
            ('rgcn_train_only.pth','rgcn_config.json','kg_normalization_stats.npz','training_summary.json','train_node_global_ids.npy'))
    config={'method':METHOD_ID,'seeds':list(args.seeds),'model_config':MODEL_CONFIG,
        'train_config':TRAIN_CONFIG,'beta_candidates':list(BETA_CANDIDATES),
        'causal_root':os.path.abspath(args.causal_root),'rgcn_root':os.path.abspath(args.rgcn_root),
        'kg_dir':os.path.abspath(args.kg_dir),'reference_path':os.path.abspath(args.reference_path),
        'script_sha256':file_sha256(__file__),
        'input_sha256':{str(p.resolve()):file_sha256(p) for p in inputs},
        'selection':'Train and validation only; all seed checkpoints and beta locked before test.',
        'test_scope':'Exploratory: this test set has already been inspected in previous experiments.',
        'prediction':'T-CLIPER + validation-selected beta times model residual in km',
        'independent_script':True}
    if METHOD_ID=='motion_conditioned_kg':
        config['refinement_config']=REFINEMENT_CONFIG
        config['sdgf_run']=os.path.abspath(args.sdgf_run) if args.sdgf_run else None
        config['scope']='ADDITIONAL frozen-SDGF fusion refinement, not a same-budget isolated gate replacement.'
        if args.sdgf_run:
            external=Path(args.sdgf_run);base_config=read_json(external/'run_config.json')
            for key in ('seeds','model_config','train_config','input_sha256','reference_path'):
                if base_config[key]!=config[key]:raise ValueError('SDGF reuse provenance mismatch: '+key)
            if base_config['method']!='sdgf_attention':raise ValueError('Expected a standalone SDGF run')
            config['sdgf_checkpoints_sha256']={str(s):file_sha256(external/f'seed_{s}'/'best_model.pth') for s in args.seeds}
    output.mkdir(parents=True,exist_ok=True)
    if previous and previous!=config:raise ValueError('Run source/input/config changed; choose a new output directory')
    if not previous:write_json(output/'run_config.json',config)
    RUN_FINGERPRINT=file_sha256(output/'run_config.json')
    print('Output:',output,flush=True)
    return output

def checked_validation(folder,own_run=True):
    path=Path(folder)/'validation.json'
    if not path.exists():return None
    row=read_json(path)
    if row['checkpoint_sha256']!=file_sha256(Path(folder)/'best_model.pth'):
        raise ValueError('Checkpoint changed: '+str(folder))
    if own_run and row['run_lock_sha256']!=RUN_FINGERPRINT:raise ValueError('Validation run lock mismatch')
    return row

def test_seed(seed,datasets,metadata,folder,motion=False,base_dir=None):
    folder=Path(folder);record=checked_validation(folder,own_run=not (motion is False and METHOD_ID=='motion_conditioned_kg'))
    if record is None:raise FileNotFoundError(folder/'validation.json')
    for key in ('correction_mean_km','correction_std_km'):
        if not np.array_equal(np.array(record[key]),metadata[key]):raise ValueError('Correction normalization mismatch')
    state=torch.load(folder/'best_model.pth',map_location=DEVICE,weights_only=True)
    if motion:
        model,_=make_motion_model(seed,metadata,base_dir)
        if state['base_checkpoint_sha256']!=file_sha256(Path(base_dir)/'best_model.pth'):
            raise ValueError('Changed refinement base checkpoint')
    else:model=PatchTSTWithRGCNChannelGated(**model_kwargs(metadata)).to(DEVICE)
    model.load_state_dict(state['model_state_dict']);loader=make_loaders(datasets,seed)['test']
    if (folder/'test_metrics.json').exists():
        old=read_json(folder/'metrics.json')
        if old['checkpoint_sha256']!=record['checkpoint_sha256']:raise ValueError('Changed tested checkpoint')
        return read_json(folder/'test_metrics.json')
    metrics=test_model(model,loader,metadata,str(folder),args_global.reference_path,record['best_beta'])
    write_json(folder/'test_metrics.json',metrics)
    write_json(folder/'metrics.json',{**record,**metrics,'test_checkpoint_sha256':record['checkpoint_sha256']})
    if motion:
        gammas=[];biases=[]
        with torch.no_grad():
            for x,k in loader:
                features=model.encode(x.to(DEVICE),k.to(DEVICE));gamma,bias=model.modulation(features[3])
                gammas.append(gamma.cpu().numpy());biases.append(bias.cpu().numpy())
        np.save(folder/'kg_modulation_gamma.npy',np.concatenate(gammas))
        np.save(folder/'kg_modulation_bias.npy',np.concatenate(biases))
    print(f"{record['method']} seed={seed} test ADE={metrics['ade_km']:.6f}, FDE={metrics['fde_km']:.6f}",flush=True)
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return metrics

def summarize_run(folder,seeds):
    rows={s:read_json(Path(folder)/f'seed_{s}'/'test_metrics.json') for s in seeds}
    keys=list(rows[seeds[0]])
    if any(set(r)!=set(keys) for r in rows.values()):raise ValueError('Metric key mismatch')
    summary={k:{'mean':float(np.mean([r[k] for r in rows.values()])),
        'std':float(np.std([r[k] for r in rows.values()],ddof=1)) if len(seeds)>1 else 0.} for k in keys}
    write_json(Path(folder)/'seed_summary.json',{'seeds':list(seeds),'metrics':summary,
        'aggregation':'Mean/sample SD of per-seed metrics, not pooled predictions.'})
    with open(Path(folder)/'all_metrics_by_seed.csv','w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['seed',*keys])
        for s,r in rows.items():w.writerow([s,*[r[k] for k in keys]])
    return summary

def compare_runs(baseline_folder,new_folder,report_folder,seeds):
    baseline_folder=Path(baseline_folder);new_folder=Path(new_folder);report_folder=Path(report_folder)
    differences={};keys=None
    for s in seeds:
        a=baseline_folder/f'seed_{s}';b=new_folder/f'seed_{s}'
        if not np.array_equal(np.load(a/'ground_truth.npy'),np.load(b/'ground_truth.npy')):
            raise ValueError('Comparison ground truth mismatch')
        with np.load(a/'test_sample_metadata.npz',allow_pickle=False) as x,np.load(b/'test_sample_metadata.npz',allow_pickle=False) as y:
            if set(x.files)!=set(y.files) or not all(np.array_equal(x[k],y[k]) for k in x.files):
                raise ValueError('Comparison sample metadata mismatch')
        differences[s]=(read_json(a/'test_metrics.json'),read_json(b/'test_metrics.json'))
        if keys is None:keys=list(differences[s][0])
        if any(set(r)!=set(keys) for r in differences[s]):raise ValueError('Comparison metric keys differ')
    rows=[];by_seed=[]
    for k in keys:
        high=('skill_vs_' in k or k.startswith('ade_skill_'))
        reference=k.startswith(('persistence_de_obs_','cliper_de_obs_'))
        a=np.array([differences[s][0][k] for s in seeds]);b=np.array([differences[s][1][k] for s in seeds])
        gains=b-a if high else a-b
        row={'metric':k,'direction':'reference' if reference else ('higher' if high else 'lower'),
            'sdgf_mean':float(a.mean()),'motion_mean':float(b.mean()),
            'sdgf_std':float(a.std(ddof=1)) if len(seeds)>1 else 0.,
            'motion_std':float(b.std(ddof=1)) if len(seeds)>1 else 0.,
            'paired_gain_mean':float(gains.mean()),
            'paired_gain_std':float(gains.std(ddof=1)) if len(seeds)>1 else 0.,
            'improvement_pct':None if high or reference or a.mean()==0 else float(gains.mean()/a.mean()*100),
            'improved_seeds':int((gains>1e-10).sum()),'worse_seeds':int((gains < -1e-10).sum()),
            'tied_seeds':int((abs(gains)<=1e-10).sum())}
        rows.append(row)
        for i,s in enumerate(seeds):by_seed.append({'seed':s,'metric':k,'sdgf':float(a[i]),'motion':float(b[i]),'gain':float(gains[i])})
    write_json(report_folder/'comparison_all_metrics.json',{'seeds':list(seeds),
        'baseline_run':str(baseline_folder.resolve()),'motion_run':str(new_folder.resolve()),
        'scope':'Exploratory; MotionKG adds a fusion-only training stage. No independent confirmation or equal-budget claim.',
        'metric_count':len(rows),'metrics':rows})
    for name,items in [('comparison_all_metrics.csv',rows),('comparison_by_seed.csv',by_seed)]:
        with open(report_folder/name,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    lines=['# SDGF vs motion-conditioned KG','',
        'All fixed seeds retained: '+', '.join(map(str,seeds))+'. Lower errors / higher skill scores are better.',
        'Positive paired gain means an improvement. All test metric fields and all 24 forecast steps are in comparison_all_metrics.csv.',
        '','| Metric | SDGF mean | MotionKG mean | Paired gain | Better seeds |',
        '|---|---:|---:|---:|---:|']
    selected=[r for r in rows if r['metric'] in OVERVIEW_METRICS]
    for r in selected:lines.append(f"| {r['metric']} | {r['sdgf_mean']:.6f} | {r['motion_mean']:.6f} | {r['paired_gain_mean']:.6f} | {r['improved_seeds']}/{len(seeds)} |")
    lines+=['','## Interpretation','',
        'MotionKG retains SDGF encoders, scoring, head and its validation-selected beta; trains only 16,928 new affine conditioning parameters.',
        'The conditioner uses relative past lon/lat, original historical speed/pressure, and differences of patch representations from the first patch.',
        'This adds a training stage and a beta-calibrated forecast loss with an anchor penalty. It is not an isolated same-budget gate ablation.',
        'Initial identity is included in validation checkpoint selection. Test metrics do not choose checkpoints or beta.',
        'The test set was previously inspected; these are exploratory reproduction results.',
        'FiLM-inspired bounded KG adaptation; not a full paper reproduction.']
    (report_folder/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('All-metric comparison:',report_folder/'COMPARISON.md',flush=True)

def pipeline(args):
    global args_global
    args_global=args;output=prepare_run(args)
    cache={}
    def data_for(seed):
        if seed not in cache:cache[seed]=load_data(os.path.join(args.causal_root,f'seed_{seed}'),seed,args.reference_path)
        return cache[seed]
    if args.phase in ('all','train-val'):
        records=[]
        for seed in args.seeds:
            data,meta=data_for(seed)
            base=output/(f'seed_{seed}' if METHOD_ID=='sdgf_attention' else f'base_sdgf/seed_{seed}')
            external=getattr(args,'sdgf_run',None)
            if METHOD_ID=='motion_conditioned_kg' and external:
                src=Path(external)/f'seed_{seed}';record=checked_validation(src,own_run=False)
                if record is None:raise FileNotFoundError(src/'validation.json')
                if not base.exists():
                    base.mkdir(parents=True)
                    for name in ('best_model.pth','validation.json','training_history.json'):
                        shutil.copy2(src/name,base/name)
                if file_sha256(base/'best_model.pth')!=file_sha256(src/'best_model.pth'):
                    raise ValueError('Copied base checkpoint differs')
            elif checked_validation(base) is None:
                if base.exists():raise ValueError('Incomplete base training directory; use a fresh output directory')
                base.mkdir(parents=True)
                train_sdgf_validation(seed,data,meta,str(base),args.reference_path)
            if METHOD_ID=='motion_conditioned_kg':
                target=output/f'seed_{seed}'
                if checked_validation(target) is None:
                    if target.exists():raise ValueError('Incomplete refinement directory; use a fresh output directory')
                    target.mkdir(parents=True)
                    train_motion_validation(seed,data,meta,target,base)
                records.append(checked_validation(target))
            else:records.append(checked_validation(base))
        write_json(output/'selection_lock.json',{'seeds':list(args.seeds),'run_lock_sha256':RUN_FINGERPRINT,
            'selected_checkpoints_sha256':{str(r['seed']):r['checkpoint_sha256'] for r in records},
            'selected_betas':{str(r['seed']):r['best_beta'] for r in records},
            'test_used_for_selection':False})
    if args.phase in ('all','evaluate-test'):
        selection=read_json(output/'selection_lock.json')
        if selection['run_lock_sha256']!=RUN_FINGERPRINT:raise ValueError('Selection lock mismatch')
        for seed in args.seeds:
            target=output/f'seed_{seed}';record=checked_validation(target)
            if selection['selected_checkpoints_sha256'][str(seed)]!=record['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed')
            if selection['selected_betas'][str(seed)]!=record['best_beta']:raise ValueError('Selected beta changed')
            data,meta=data_for(seed)
            if METHOD_ID=='motion_conditioned_kg':
                base=output/'base_sdgf'/f'seed_{seed}'
                test_seed(seed,data,meta,base)
                test_seed(seed,data,meta,target,motion=True,base_dir=base)
            else:test_seed(seed,data,meta,target)
        summarize_run(output,args.seeds)
        if METHOD_ID=='motion_conditioned_kg':
            summarize_run(output/'base_sdgf',args.seeds)
            compare_runs(args.compare_with or args.sdgf_run or output/'base_sdgf',output,output,args.seeds)
        elif args.compare_with:compare_runs(output,args.compare_with,output,args.seeds)
    print('Completed phase',args.phase,'in',output,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    parser.add_argument('--phase',choices=('all','train-val','evaluate-test'),default='all')
    parser.add_argument('--causal-root',default=os.path.join(DEFAULT_UPSTREAM,'causal_embeddings'))
    parser.add_argument('--rgcn-root',default=os.path.join(DEFAULT_UPSTREAM,'rgcn'))
    parser.add_argument('--kg-dir',default=os.path.join(SCRIPT_DIR,'data','kg_enriched_v1'))
    parser.add_argument('--reference-path',default=DEFAULT_REFERENCE_NPZ)
    parser.add_argument('--output-dir')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--compare-with',help='Optional other standalone result directory; does not execute any script')
    if METHOD_ID=='motion_conditioned_kg':
        parser.add_argument('--sdgf-run',help='Optional compatible SDGF result directory. Omit to train own SDGF base independently.')
    pipeline(parser.parse_args())
