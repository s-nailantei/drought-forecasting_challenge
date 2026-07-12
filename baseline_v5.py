"""
v5: 1200 trees + spatial features + more lags
Target: beat 0.3582
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

all_times = sorted(train['time'].unique())
time_idx = {t: i for i, t in enumerate(all_times)}
train['time_idx'] = train['time'].map(time_idx)
max_idx = max(time_idx.values())
test['time_idx'] = test['time'].apply(lambda t: time_idx.get(t, max_idx + 1))

# LAG FEATURES
print("Building lags...")
for lp in [1, 2, 3, 6]:
    lag = train[['lat','lon','time_idx','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']].copy()
    lag['time_idx'] += lp
    lag['lag_key'] = lag['lat'].astype(str) + '_' + lag['lon'].astype(str) + '_' + lag['time_idx'].astype(str)
    lag = lag[['lag_key','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']].rename(columns={
        'TWS_t': f'TWS_t_lag{lp}', 'SOIL_MOISTURE_t': f'SOIL_MOISTURE_t_lag{lp}', 'SPEI_01_t': f'SPEI_01_t_lag{lp}'
    })
    train['lag_key'] = train['lat'].astype(str) + '_' + train['lon'].astype(str) + '_' + (train['time_idx'] - lp).astype(str)
    test['lag_key'] = test['lat'].astype(str) + '_' + test['lon'].astype(str) + '_' + (test['time_idx'] - lp).astype(str)
    train = train.merge(lag, on='lag_key', how='left')
    test = test.merge(lag, on='lag_key', how='left')

# Fallback for test NaNs
last_t = train[train['time_idx'] == train['time_idx'].max()][['lat','lon','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']]
last_t.columns = ['lat','lon','TWS_lf','SOIL_lf','SPEI_lf']
test = test.merge(last_t, on=['lat','lon'], how='left')
for lp in [1,2,3,6]:
    for c in ['TWS_t','SOIL_MOISTURE_t','SPEI_01_t']:
        col = f'{c}_lag{lp}'
        if col in test.columns:
            test[col] = test[col].fillna(test[f'{c.replace("t","").replace("_","_")}lf' if False else {
                'TWS_t': 'TWS_lf', 'SOIL_MOISTURE_t': 'SOIL_lf', 'SPEI_01_t': 'SPEI_lf'
            }.get(c, 'TWS_lf')])

before = len(train)
train = train.dropna(subset=['TWS_t_lag1']).reset_index(drop=True)
print(f"Train: {len(train)} (dropped {before - len(train)})")

# SPATIAL FEATURES
print("Spatial features...")
for df in [train, test]:
    df['abs_lat'] = df['lat'].abs()
    df['lat_round'] = df['lat'].round(0)

regional = train.groupby(['month', 'lat_round']).agg(
    tws_reg_mean=('TWS_t', 'mean'), tws_reg_std=('TWS_t', 'std'),
    soil_reg_mean=('SOIL_MOISTURE_t', 'mean'),
).reset_index()
train = train.merge(regional, on=['month', 'lat_round'], how='left')
test = test.merge(regional, on=['month', 'lat_round'], how='left')

# FEATURE ENGINEERING
print("Features...")
for df in [train, test]:
    df['spei_mean'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].mean(axis=1)
    df['spei_std'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].std(axis=1)
    df['spei_range'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].max(axis=1) - df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].min(axis=1)
    df['tws_soil'] = df['TWS_t'] * df['SOIL_MOISTURE_t']
    df['tws_spei01'] = df['TWS_t'] * df['SPEI_01_t']
    df['tws_lag1_diff'] = df['TWS_t'] - df['TWS_t_lag1']
    df['tws_lag3_diff'] = df['TWS_t'] - df['TWS_t_lag3']
    df['soil_lag1_diff'] = df['SOIL_MOISTURE_t'] - df['SOIL_MOISTURE_t_lag1']
    df['tws_trend_1_3'] = df['TWS_t_lag1'] - df['TWS_t_lag3']
    df['tws_trend_1_6'] = df['TWS_t_lag1'] - df['TWS_t_lag6']

features = [
    'TWS_t', 'SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t',
    'month_sin', 'month_cos', 'lat', 'lon', 'abs_lat', 'month',
    'spei_mean', 'spei_std', 'spei_range', 'tws_soil', 'tws_spei01',
    'TWS_t_lag1', 'SOIL_MOISTURE_t_lag1', 'SPEI_01_t_lag1',
    'TWS_t_lag2', 'TWS_t_lag3', 'TWS_t_lag6',
    'tws_lag1_diff', 'tws_lag3_diff', 'soil_lag1_diff',
    'tws_trend_1_3', 'tws_trend_1_6',
    'tws_reg_mean', 'tws_reg_std', 'soil_reg_mean',
]

X = train[features].values
y = train['target'].values
Xt = test[features].values
print(f"Features: {len(features)}, X: {X.shape}")

lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 255, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,     'n_estimators': 1000,
    'verbose': -1, 'seed': SEED, 'n_jobs': -1, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
}

print("\nTraining 3-Fold CV...")
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
test_preds = np.zeros(len(Xt))
scores = []

for fold, (tri, vai) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/3...")
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X[tri], y[tri], eval_set=[(X[vai], y[vai])],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    vp = m.predict(X[vai])
    test_preds += m.predict(Xt) / 3
    rmse = np.sqrt(mean_squared_error(y[vai], vp))
    scores.append(rmse)
    print(f"  RMSE: {rmse:.6f}")

print(f"\nOOF RMSE: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")
print(f"v4 was 0.3582")

sub = sample_sub.copy()
sub['Target'] = test_preds
sub.to_csv('submission_v5.csv', index=False)
print(f"Saved submission_v5.csv, Time: {time.time()-t0:.0f}s")

imp = pd.DataFrame({'f': features, 'i': m.feature_importances_}).sort_values('i', ascending=False)
print("\nTop features:")
print(imp.head(15).to_string(index=False))
