"""
v3 FAST: Lag via merge + faster model (800 trees)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings, time
warnings.filterwarnings('ignore')

SEED = 42
t0 = time.time()

print("Loading data...")
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sample_sub = pd.read_csv('SampleSubmission.csv')

train['time'] = pd.to_datetime(train['time'])
test['time'] = pd.to_datetime(test['time'])
train['month'] = train['time'].dt.month
test['month'] = test['time'].dt.month
print(f"Load: {time.time()-t0:.1f}s")

# Time index
all_times = sorted(train['time'].unique())
time_idx = {t: i for i, t in enumerate(all_times)}
train['time_idx'] = train['time'].map(time_idx)
max_idx = max(time_idx.values())
test['time_idx'] = test['time'].apply(lambda t: time_idx.get(t, max_idx + 1))

# Lag merge (fast)
print("Lag features...")
train['lag_key'] = train['lat'].astype(str) + '_' + train['lon'].astype(str) + '_' + (train['time_idx'] - 1).astype(str)
lag = train[['lat','lon','time_idx','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']].copy()
lag['time_idx'] += 1
lag['lag_key'] = lag['lat'].astype(str) + '_' + lag['lon'].astype(str) + '_' + lag['time_idx'].astype(str)
lag = lag[['lag_key','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']].rename(columns={'TWS_t':'TWS_t_lag1','SOIL_MOISTURE_t':'SOIL_MOISTURE_t_lag1','SPEI_01_t':'SPEI_01_t_lag1'})

train = train.merge(lag, on='lag_key', how='left')
test['lag_key'] = test['lat'].astype(str) + '_' + test['lon'].astype(str) + '_' + (test['time_idx'] - 1).astype(str)
test = test.merge(lag, on='lag_key', how='left')

# Fallback for test NaN: use last train period values
last_train = train[train['time_idx'] == train['time_idx'].max()][['lat','lon','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']]
last_train.columns = ['lat','lon','TWS_t_lag1_f','SOIL_MOISTURE_t_lag1_f','SPEI_01_t_lag1_f']
test = test.merge(last_train, on=['lat','lon'], how='left')
for c in ['TWS_t_lag1','SOIL_MOISTURE_t_lag1','SPEI_01_t_lag1']:
    fc = c + '_f'
    if fc in test.columns:
        test[c] = test[c].fillna(test[fc])

# Drop train NaN lags
before = len(train)
train = train.dropna(subset=['TWS_t_lag1']).reset_index(drop=True)
print(f"Dropped {before - len(train)} train rows, kept {len(train)}")

# Features
print("Features...")
for df in [train, test]:
    df['abs_lat'] = df['lat'].abs()
    df['spei_mean'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].mean(axis=1)
    df['spei_std'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].std(axis=1)
    df['tws_soil'] = df['TWS_t'] * df['SOIL_MOISTURE_t']
    df['tws_spei01'] = df['TWS_t'] * df['SPEI_01_t']
    df['tws_lag1_diff'] = df['TWS_t'] - df['TWS_t_lag1']

features = [
    'TWS_t', 'SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t',
    'month_sin', 'month_cos', 'lat', 'lon', 'abs_lat', 'month',
    'spei_mean', 'spei_std', 'tws_soil', 'tws_spei01',
    'TWS_t_lag1', 'SOIL_MOISTURE_t_lag1', 'SPEI_01_t_lag1', 'tws_lag1_diff',
]

X = train[features].values
y = train['target'].values
Xt = test[features].values
print(f"Features: {len(features)}, X: {X.shape}")

# Model
lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 255, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'n_estimators': 800,
    'verbose': -1, 'seed': SEED, 'n_jobs': -1, 'min_child_samples': 50,
}

print("\nTraining 3-Fold CV...")
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
test_preds = np.zeros(len(Xt))
scores = []

for fold, (tri, vai) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/3...")
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X[tri], y[tri], eval_set=[(X[vai], y[vai])],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    vp = m.predict(X[vai])
    test_preds += m.predict(Xt) / 3
    rmse = np.sqrt(mean_squared_error(y[vai], vp))
    scores.append(rmse)
    print(f"  RMSE: {rmse:.6f}")

print(f"\nOOF RMSE: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")
print(f"v2 was 0.4363")

sub = sample_sub.copy()
sub['Target'] = test_preds
sub.to_csv('submission_v3.csv', index=False)
print(f"Saved submission_v3.csv, Time: {time.time()-t0:.0f}s")
