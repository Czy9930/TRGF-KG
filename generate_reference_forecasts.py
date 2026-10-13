import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
import pickle

import numpy as np


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
SOURCE_URL = 'https://repository.library.noaa.gov/view/noaa/47867/noaa_47867_DS1.pdf'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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


def load_split(split, metadata, input_dir=INPUT_DIR):
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


def generate(output_path, input_dir=INPUT_DIR):
    input_dir = os.path.abspath(input_dir)
    with open(os.path.join(input_dir, 'metadata.pkl'), 'rb') as file:
        metadata = pickle.load(file)
    if (metadata['input_features'] != ['lng', 'lat', 'speed', 'pressure']
            or metadata['target_features'] != ['lng', 'lat']
            or metadata['lookback_window'] != LOOKBACK
            or metadata['forecast_horizon'] != HORIZON
            or metadata['n_input_features'] != 4
            or metadata['n_target_features'] != 2
            or metadata['scaler_X'].n_features_in_ != 4
            or metadata['scaler_y'].n_features_in_ != 2):
        raise ValueError('特征工程 metadata 与 12×4 → 24×2 任务不一致')
    train = load_split('train', metadata, input_dir)
    test = load_split('test', metadata, input_dir)
    if not np.all((test['years'] >= 2021) & (test['years'] <= 2022)):
        raise ValueError('test 集年份异常')
    if set(map(str, train['typhoon_ids'])) & set(map(str, test['typhoon_ids'])):
        raise ValueError('训练与测试台风发生重叠')
    velocities, velocity_times, storm_count, unique_count = (
        unique_training_observations(train)
    )
    fields, monthly_counts = build_climatology(velocities, velocity_times)
    n = len(test['X'])
    persistence = np.empty((n, HORIZON, 2), dtype=np.float64)
    cliper = np.empty_like(persistence)
    chosen_steps = np.empty(n, dtype=np.int32)
    chosen_times = np.empty(n, dtype=test['input_times'].dtype)
    chosen_coordinates = np.empty((n, 2), dtype=np.float64)
    actual_lags = np.empty(n, dtype=np.float64)
    for i in range(n):
        velocity, chosen_steps[i], actual_lags[i] = initial_motion(
            test['input_coordinates'][i], test['input_times'][i],
            test['input_time_steps'][i],
        )
        chosen_index = int(np.flatnonzero(
            test['input_time_steps'][i] == chosen_steps[i]
        )[0])
        chosen_times[i] = test['input_times'][i, chosen_index]
        chosen_coordinates[i] = test['input_coordinates'][i, chosen_index]
        initial = test['input_coordinates'][i, -1]
        targets = test['forecast_elapsed_hours'][i]
        origin_time = test['forecast_origin_time'][i]
        persistence[i] = integrate(initial, velocity, origin_time, targets, None)
        cliper[i] = integrate(initial, velocity, origin_time, targets, fields)
        if (i + 1) % 100 == 0:
            print(f'已生成参考轨迹 {i + 1}/{n}', flush=True)
    if (not np.isfinite(persistence).all() or not np.isfinite(cliper).all()
            or persistence.shape != test['ground_truth'].shape):
        raise ValueError('参考预报输出形状或数值异常')
    persistence_definition = (
        'Constant initial zonal/meridional motion (km/h); initial vector from '
        'the historical observation nearest to 12 h before origin; actual '
        'selected lag saved per sample; 1 h trajectory integration at actual leads.'
    )
    cliper_definition = (
        'DeMaria et al. (2022) J-North track equations with Tu=70 h, Tv=46 h; '
        'two-pass Barnes training-only monthly motion field: RoI/EFD '
        '6250/3375 km then 1875/1250 km; 2-degree 30E-220E, 0N-70N grid; '
        'empty grid cells use the monthly mean in pass 1 and zero correction '
        'in pass 2; out-of-grid queries use the nearest grid edge; '
        'monthly fields interpolated by calendar date; 1 h Adams-Bashforth-2 '
        'integration to actual target timestamps. Fields reconstructed from '
        'deduplicated 2000-2018 training storms, not operational JTWC fields.'
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary = output_path + '.partial.npz'
    np.savez_compressed(
        temporary,
        persistence_predictions=persistence,
        cliper_predictions=cliper,
        ground_truth=test['ground_truth'],
        persistence_definition=np.array(persistence_definition),
        cliper_definition=np.array(cliper_definition),
        initial_motion_reference_time_steps=chosen_steps,
        initial_motion_reference_times=chosen_times,
        initial_motion_reference_coordinates=chosen_coordinates,
        initial_motion_reference_lag_hours=actual_lags,
        **{key: test[key] for key in ALIGNMENT_KEYS},
    )
    os.replace(temporary, output_path)
    summary = {
        'source': SOURCE_URL,
        'train_samples': len(train['X']),
        'train_typhoons': storm_count,
        'unique_train_observations': unique_count,
        'climate_motion_records': len(velocities),
        'monthly_motion_records': monthly_counts,
        'test_samples': n,
        'initial_lag_min_h': float(actual_lags.min()),
        'initial_lag_median_h': float(np.median(actual_lags)),
        'initial_lag_max_h': float(actual_lags.max()),
        'persistence_definition': persistence_definition,
        'cliper_definition': cliper_definition,
        'input_dir': input_dir,
        'train_data_sha256': sha256(os.path.join(input_dir, 'train_data.npz')),
        'test_data_sha256': sha256(os.path.join(input_dir, 'test_data.npz')),
        'metadata_sha256': sha256(os.path.join(input_dir, 'metadata.pkl')),
        'generator_sha256': sha256(__file__),
        'reference_npz_sha256': sha256(output_path),
    }
    with open(os.path.join(os.path.dirname(output_path),
                           'reference_generation_summary.json'),
              'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f'参考预报已保存: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成训练期气候场的 Persistence/T-CLIPER 参考轨迹')
    parser.add_argument('--output', default=OUTPUT_PATH)
    parser.add_argument('--input-dir', default=INPUT_DIR)
    args = parser.parse_args()
    generate(os.path.abspath(args.output), args.input_dir)
