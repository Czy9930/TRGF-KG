import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'input_data', 'past_2000-2022typhoons')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'input')
KG_MAPPING_PATH = os.path.join(
    SCRIPT_DIR, 'data', 'temporal_kg', 'entity_time_mapping.csv'
)

INPUT_FEATURES = ['lng', 'lat', 'speed', 'pressure']
TARGET_FEATURES = ['lng', 'lat']
REQUIRED_FEATURES = INPUT_FEATURES

LOOKBACK_WINDOW = 12
FORECAST_HORIZON = 24

TRAIN_YEARS = (2000, 2018)
VAL_YEARS = (2019, 2020)
TEST_YEARS = (2021, 2022)


def load_raw_data():
    print("=" * 80)
    print("步骤1: 数据加载（长预测时域模式，forecast=24）")
    print("=" * 80)

    all_typhoons = []
    typhoon_files = [f for f in os.listdir(DATA_DIR) if not f.endswith('.csv')]

    print(f"发现 {len(typhoon_files)} 个台风文件")

    skipped_short = 0
    skipped_raw_short = 0
    skipped_after_dropna = 0
    skipped_missing_columns = 0
    dropped_missing_input_points = 0

    for filename in typhoon_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            year_suffix = int(filename[:2])
            year = 2000 + year_suffix
            full_df = pd.read_csv(filepath, encoding='utf-8')

            if not all(col in full_df.columns for col in REQUIRED_FEATURES + ['OBS', 'time']):
                skipped_missing_columns += 1
                continue

            typhoon_name = ""
            if 'name' in full_df.columns and len(full_df) > 0:
                typhoon_name = str(full_df['name'].iloc[0]).strip()

            typhoon_id = filename + typhoon_name

            df = full_df[INPUT_FEATURES].copy()
            df = df.dropna(subset=REQUIRED_FEATURES)
            dropped_missing_input_points += len(full_df) - len(df)


            obs_steps = pd.to_numeric(
                full_df.loc[df.index, 'OBS'].astype(str).str.strip().str.extract(r'^OBS_(\d+)$', expand=False),
                errors='coerce')
            if obs_steps.isna().any() or obs_steps.duplicated().any() or not obs_steps.is_monotonic_increasing:
                raise ValueError('原始 OBS 编号缺失、重复或顺序异常')
            df['kg_time_step'] = obs_steps.astype(np.int32)


            observation_times = pd.to_datetime(
                full_df.loc[df.index, 'time'], errors='coerce'
            )
            if observation_times.isna().any():
                raise ValueError('有效观测存在无法解析的原始 time')
            time_values = observation_times.to_numpy(dtype='datetime64[ns]')
            if not np.all(np.diff(time_values) > np.timedelta64(0, 'ns')):
                raise ValueError('有效观测的原始 time 不是严格递增')
            df['observation_time'] = time_values

            min_length = LOOKBACK_WINDOW + FORECAST_HORIZON
            if len(df) >= min_length:
                all_typhoons.append({
                    'name': typhoon_id,
                    'year': year,
                    'data': df[INPUT_FEATURES + ['kg_time_step', 'observation_time']].reset_index(drop=True)
                })
            else:
                skipped_short += 1
                if len(full_df) < min_length:
                    skipped_raw_short += 1
                else:
                    skipped_after_dropna += 1
        except Exception as e:
            print(f"  跳过 {filename}: {str(e)}")
            continue

    print(f"\n✓ 成功加载 {len(all_typhoons)} 个台风")
    print(f"  因长度不足跳过: {skipped_short} 个（需要至少 {LOOKBACK_WINDOW + FORECAST_HORIZON} 步）")
    print(f"    原始观测不足: {skipped_raw_short} 个；清洗缺失值后不足: {skipped_after_dropna} 个")
    print(f"  缺少必要输入列、OBS 或 time 列跳过: {skipped_missing_columns} 个")
    print(f"  因实际输入变量缺失删除观测点: {dropped_missing_input_points} 个")
    total_samples = sum(len(t['data']) for t in all_typhoons)
    print(f"  总观测点数: {total_samples}")
    print(f"  输入特征: {INPUT_FEATURES}")
    print(f"  时序窗口: lookback={LOOKBACK_WINDOW}, forecast={FORECAST_HORIZON}")

    return all_typhoons


def create_sequences(typhoons_data):
    print("\n" + "=" * 80)
    print("步骤2: 构建序列（forecast=24）")
    print("=" * 80)

    X_list, y_list = [], []
    sample_info_list = []

    for typhoon in typhoons_data:
        typhoon_id = typhoon['name']
        typhoon_year = typhoon['year']
        X_data = typhoon['data'][INPUT_FEATURES].values
        y_data = typhoon['data'][TARGET_FEATURES].values
        kg_time_steps = typhoon['data']['kg_time_step'].to_numpy()
        observation_times = typhoon['data']['observation_time'].to_numpy(
            dtype='datetime64[ns]'
        )

        for i in range(len(X_data) - LOOKBACK_WINDOW - FORECAST_HORIZON + 1):
            origin_time = observation_times[i + LOOKBACK_WINDOW - 1]
            forecast_times = observation_times[
                i + LOOKBACK_WINDOW:i + LOOKBACK_WINDOW + FORECAST_HORIZON
            ].copy()
            elapsed_hours = (
                (forecast_times - origin_time) / np.timedelta64(1, 'h')
            ).astype(np.float64)
            if (not np.isfinite(elapsed_hours).all()
                    or elapsed_hours[0] <= 0
                    or not np.all(np.diff(elapsed_hours) > 0)):
                raise ValueError(
                    f'{typhoon_id} start_index={i}: '
                    'forecast_elapsed_hours 必须为正且严格递增'
                )
            X = X_data[i:i + LOOKBACK_WINDOW]
            y = y_data[i + LOOKBACK_WINDOW:i + LOOKBACK_WINDOW + FORECAST_HORIZON]
            X_list.append(X)
            y_list.append(y)
            sample_info_list.append({
                'typhoon_id': typhoon_id,
                'year': typhoon_year,
                'start_index': i,

                'forecast_origin_index': i + LOOKBACK_WINDOW - 1,
                'forecast_start_index': i + LOOKBACK_WINDOW,
                'forecast_end_index': i + LOOKBACK_WINDOW + FORECAST_HORIZON - 1,

                'input_time_steps': kg_time_steps[i:i + LOOKBACK_WINDOW].astype(np.int32),
                'forecast_time_steps': kg_time_steps[
                    i + LOOKBACK_WINDOW:i + LOOKBACK_WINDOW + FORECAST_HORIZON
                ].astype(np.int32),
                'forecast_origin_time_step': int(kg_time_steps[i + LOOKBACK_WINDOW - 1]),
                'forecast_start_time_step': int(kg_time_steps[i + LOOKBACK_WINDOW]),
                'forecast_end_time_step': int(kg_time_steps[i + LOOKBACK_WINDOW + FORECAST_HORIZON - 1]),
                'input_times': observation_times[i:i + LOOKBACK_WINDOW].copy(),
                'forecast_times': forecast_times,
                'forecast_origin_time': origin_time,
                'forecast_elapsed_hours': elapsed_hours,
            })

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"✓ 序列构建完成")
    print(f"  总样本数: {len(X)}")
    print(f"  输入形状: {X.shape}  (samples, lookback={LOOKBACK_WINDOW}, features={len(INPUT_FEATURES)})")
    print(f"  输出形状: {y.shape}  (samples, forecast={FORECAST_HORIZON}, targets={len(TARGET_FEATURES)})")

    return X, y, sample_info_list


def validate_with_kg(sample_info):
    print("\n" + "=" * 80)
    print("步骤3: 检查轨迹样本与知识图谱时间索引一致性")
    print("=" * 80)

    df_kg = pd.read_csv(
        KG_MAPPING_PATH,
        encoding='utf-8-sig',
        dtype={'typhoon_id': str},
    )
    required_cols = {'typhoon_id', 'time_step'}
    if missing_cols := required_cols - set(df_kg.columns):
        raise ValueError(f"KG 实体时间映射缺少字段: {sorted(missing_cols)}")
    if df_kg['typhoon_id'].isna().any() or df_kg['time_step'].isna().any():
        raise ValueError("KG 实体时间映射存在空 typhoon_id 或 time_step")
    df_kg['time_step'] = pd.to_numeric(
        df_kg['time_step'], errors='raise'
    ).astype(np.int64)
    kg_pairs = set(zip(df_kg['typhoon_id'], df_kg['time_step']))
    if len(kg_pairs) != len(df_kg):
        raise ValueError("KG 中同一台风的 time_step 对应多个 OBS")

    missing = []
    for info in sample_info:
        typhoon_id = str(info['typhoon_id'])
        for kind, time_steps in (
            ('input', info['input_time_steps']),
            ('forecast', info['forecast_time_steps']),
            ('origin', [info['forecast_origin_time_step']]),
        ):
            for time_step in time_steps:
                pair = (typhoon_id, int(time_step))
                if pair not in kg_pairs:
                    missing.append((*pair, kind))
    if missing:
        print(f"发现 {len(missing)} 个轨迹-KG时间索引无法匹配")
        print("示例:", missing[:20])
        raise ValueError("轨迹数据与KG的time_step存在不一致")
    print(f"✓ 全部 {len(sample_info)} 个样本与KG time_step正确对齐")


def split_by_time_and_normalize(X, y, sample_info):
    print("\n" + "=" * 80)
    print("步骤4: 按时间划分数据集与归一化")
    print("=" * 80)

    train_indices, val_indices, test_indices = [], [], []

    for idx, info in enumerate(sample_info):
        year = info['year']
        if TRAIN_YEARS[0] <= year <= TRAIN_YEARS[1]:
            train_indices.append(idx)
        elif VAL_YEARS[0] <= year <= VAL_YEARS[1]:
            val_indices.append(idx)
        elif TEST_YEARS[0] <= year <= TEST_YEARS[1]:
            test_indices.append(idx)

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    sample_info_train = [sample_info[i] for i in train_indices]
    sample_info_val = [sample_info[i] for i in val_indices]
    sample_info_test = [sample_info[i] for i in test_indices]

    print(f"✓ 数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    n_samples, n_timesteps, n_input_features = X_train.shape
    _, _, n_target_features = y_train.shape

    X_train_2d = X_train.reshape(-1, n_input_features)
    scaler_X.fit(X_train_2d)

    X_train = scaler_X.transform(X_train_2d).reshape(n_samples, n_timesteps, n_input_features)
    X_val = scaler_X.transform(X_val.reshape(-1, n_input_features)).reshape(len(X_val), n_timesteps, n_input_features)
    X_test = scaler_X.transform(X_test.reshape(-1, n_input_features)).reshape(len(X_test), n_timesteps, n_input_features)

    y_train_2d = y_train.reshape(-1, n_target_features)
    scaler_y.fit(y_train_2d)

    y_train = scaler_y.transform(y_train_2d).reshape(len(y_train), FORECAST_HORIZON, n_target_features)
    y_val = scaler_y.transform(y_val.reshape(-1, n_target_features)).reshape(len(y_val), FORECAST_HORIZON, n_target_features)
    y_test = scaler_y.transform(y_test.reshape(-1, n_target_features)).reshape(len(y_test), FORECAST_HORIZON, n_target_features)

    print(f"✓ 归一化完成（仅用训练集拟合scaler）")

    return (X_train, y_train, sample_info_train), (X_val, y_val, sample_info_val), (X_test, y_test, sample_info_test), (scaler_X, scaler_y)


def save_processed_data(train_data, val_data, test_data, scalers):
    print("\n" + "=" * 80)
    print("步骤5: 保存处理后的数据")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scaler_X, scaler_y = scalers
    X_train, y_train, sample_info_train = train_data
    X_val, y_val, sample_info_val = val_data
    X_test, y_test, sample_info_test = test_data

    def convert_sample_info_to_arrays(sample_info_list):
        typhoon_ids = np.array([s['typhoon_id'] for s in sample_info_list], dtype='U100')
        start_indices = np.array([s['start_index'] for s in sample_info_list], dtype=np.int32)
        forecast_origin_indices = np.array(
            [s['forecast_origin_index'] for s in sample_info_list], dtype=np.int32)
        forecast_start_indices = np.array(
            [s['forecast_start_index'] for s in sample_info_list], dtype=np.int32)
        forecast_end_indices = np.array(
            [s['forecast_end_index'] for s in sample_info_list], dtype=np.int32)
        input_time_steps = np.stack(
            [s['input_time_steps'] for s in sample_info_list], axis=0).astype(np.int32)
        forecast_time_steps = np.stack(
            [s['forecast_time_steps'] for s in sample_info_list], axis=0).astype(np.int32)
        forecast_origin_time_steps = np.array(
            [s['forecast_origin_time_step'] for s in sample_info_list], dtype=np.int32)
        forecast_start_time_steps = np.array(
            [s['forecast_start_time_step'] for s in sample_info_list], dtype=np.int32)
        forecast_end_time_steps = np.array(
            [s['forecast_end_time_step'] for s in sample_info_list], dtype=np.int32)
        input_times = np.stack(
            [s['input_times'] for s in sample_info_list], axis=0
        ).astype('datetime64[ns]')
        forecast_times = np.stack(
            [s['forecast_times'] for s in sample_info_list], axis=0
        ).astype('datetime64[ns]')
        forecast_origin_time = np.array(
            [s['forecast_origin_time'] for s in sample_info_list],
            dtype='datetime64[ns]',
        )
        forecast_elapsed_hours = np.stack(
            [s['forecast_elapsed_hours'] for s in sample_info_list], axis=0
        ).astype(np.float64)
        years = np.array([s['year'] for s in sample_info_list], dtype=np.int32)
        return (typhoon_ids, start_indices, forecast_origin_indices,
                forecast_start_indices, forecast_end_indices,
                input_time_steps, forecast_time_steps,
                forecast_origin_time_steps, forecast_start_time_steps,
                forecast_end_time_steps, input_times, forecast_times,
                forecast_origin_time, forecast_elapsed_hours, years)

    (train_ids, train_starts, train_origins, train_forecast_starts,
     train_forecast_ends, train_input_time_steps, train_forecast_time_steps,
     train_origin_steps, train_forecast_start_steps,
     train_forecast_end_steps, train_input_times, train_forecast_times,
     train_origin_time, train_elapsed_hours,
     train_years) = convert_sample_info_to_arrays(sample_info_train)
    (val_ids, val_starts, val_origins, val_forecast_starts,
     val_forecast_ends, val_input_time_steps, val_forecast_time_steps,
     val_origin_steps, val_forecast_start_steps,
     val_forecast_end_steps, val_input_times, val_forecast_times,
     val_origin_time, val_elapsed_hours,
     val_years) = convert_sample_info_to_arrays(sample_info_val)
    (test_ids, test_starts, test_origins, test_forecast_starts,
     test_forecast_ends, test_input_time_steps, test_forecast_time_steps,
     test_origin_steps, test_forecast_start_steps,
     test_forecast_end_steps, test_input_times, test_forecast_times,
     test_origin_time, test_elapsed_hours,
     test_years) = convert_sample_info_to_arrays(sample_info_test)

    np.savez_compressed(os.path.join(OUTPUT_DIR, 'train_data.npz'),
                        X=X_train, y=y_train,
                        typhoon_ids=train_ids, start_indices=train_starts,
                        forecast_origin_indices=train_origins,
                        forecast_start_indices=train_forecast_starts,
                        forecast_end_indices=train_forecast_ends,
                        input_time_steps=train_input_time_steps,
                        forecast_time_steps=train_forecast_time_steps,
                        forecast_origin_time_steps=train_origin_steps,
                        forecast_start_time_steps=train_forecast_start_steps,
                        forecast_end_time_steps=train_forecast_end_steps,
                        input_times=train_input_times,
                        forecast_times=train_forecast_times,
                        forecast_origin_time=train_origin_time,
                        forecast_elapsed_hours=train_elapsed_hours,
                        years=train_years)
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'val_data.npz'),
                        X=X_val, y=y_val,
                        typhoon_ids=val_ids, start_indices=val_starts,
                        forecast_origin_indices=val_origins,
                        forecast_start_indices=val_forecast_starts,
                        forecast_end_indices=val_forecast_ends,
                        input_time_steps=val_input_time_steps,
                        forecast_time_steps=val_forecast_time_steps,
                        forecast_origin_time_steps=val_origin_steps,
                        forecast_start_time_steps=val_forecast_start_steps,
                        forecast_end_time_steps=val_forecast_end_steps,
                        input_times=val_input_times,
                        forecast_times=val_forecast_times,
                        forecast_origin_time=val_origin_time,
                        forecast_elapsed_hours=val_elapsed_hours,
                        years=val_years)
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'test_data.npz'),
                        X=X_test, y=y_test,
                        typhoon_ids=test_ids, start_indices=test_starts,
                        forecast_origin_indices=test_origins,
                        forecast_start_indices=test_forecast_starts,
                        forecast_end_indices=test_forecast_ends,
                        input_time_steps=test_input_time_steps,
                        forecast_time_steps=test_forecast_time_steps,
                        forecast_origin_time_steps=test_origin_steps,
                        forecast_start_time_steps=test_forecast_start_steps,
                        forecast_end_time_steps=test_forecast_end_steps,
                        input_times=test_input_times,
                        forecast_times=test_forecast_times,
                        forecast_origin_time=test_origin_time,
                        forecast_elapsed_hours=test_elapsed_hours,
                        years=test_years)

    metadata = {
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'lookback_window': LOOKBACK_WINDOW,
        'forecast_horizon': FORECAST_HORIZON,
        'input_features': INPUT_FEATURES,
        'target_features': TARGET_FEATURES,
        'n_input_features': len(INPUT_FEATURES),
        'n_target_features': len(TARGET_FEATURES),
        'train_years': TRAIN_YEARS,
        'val_years': VAL_YEARS,
        'test_years': TEST_YEARS,
        'kg_time_step_source': '原始 CSV 的 OBS_n 编号（n 为 KG time_step）',
        'observation_time_source': '原始 CSV 的 time 列（严格递增，不要求固定时间间隔）',
        'forecast_elapsed_hours_definition': 'forecast_times - forecast_origin_time，以小时计',
    }

    with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    df_features = pd.DataFrame({
        'feature_type': ['input'] * len(INPUT_FEATURES) + ['target'] * len(TARGET_FEATURES),
        'feature_name': INPUT_FEATURES + TARGET_FEATURES
    })
    df_features.to_csv(os.path.join(OUTPUT_DIR, 'feature_list.csv'), index=False, encoding='utf-8-sig')

    manifest_records = []
    for split_name, sample_info_list in [
        ('train', sample_info_train),
        ('val', sample_info_val),
        ('test', sample_info_test),
    ]:
        seen = set()
        for info in sample_info_list:
            typhoon_id = info['typhoon_id']
            if typhoon_id in seen:
                continue
            seen.add(typhoon_id)
            manifest_records.append({
                'typhoon_id': typhoon_id,
                'year': info['year'],
                'split': split_name,
            })

    df_manifest = pd.DataFrame(manifest_records, columns=['typhoon_id', 'year', 'split'])
    df_manifest.to_csv(os.path.join(OUTPUT_DIR, 'typhoon_split_manifest.csv'),
                       index=False, encoding='utf-8-sig')
    print("✓ 已保存台风数据划分清单")

    print(f"✓ 数据已保存到: {OUTPUT_DIR}")


def main():
    print("=" * 80)
    print("长预测时域（24步）台风特征工程")
    print("=" * 80)
    print(f"\n★ forecast_horizon=24（预测未来24个观测步；实际时长依原始时间戳而定）")
    print(f"  输入特征: {INPUT_FEATURES}")
    print(f"  lookback={LOOKBACK_WINDOW}, forecast={FORECAST_HORIZON}")

    all_typhoons = load_raw_data()
    X, y, sample_info = create_sequences(all_typhoons)
    validate_with_kg(sample_info)
    train_data, val_data, test_data, scalers = split_by_time_and_normalize(X, y, sample_info)
    save_processed_data(train_data, val_data, test_data, scalers)

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()
