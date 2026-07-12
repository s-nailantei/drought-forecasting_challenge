"""
v6: Fix predictions for masked test rows
Strategy: use v4 model, but replace predictions for masked rows with lag-based estimates
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
print(f"Test masked: {test['TWS_t_masked'].sum()}/{len(test)} ({test['TWS_t_masked'].mean()*100:.0f}%)")

all_times = sorted(train['time'].unique())
time_idx = {t: i for i, t in enumerate(all_times)}
train['time_idx'] = train['time'].map(time_idx)
max_idx = max(time_idx.values())
test['time_idx'] = test['time'].apply(lambda t: time_idx.get(t, max_idx + 1))

# LAGS
print("Lags...")
for lp in [1, 2, 3, 6]:
    lag = train[['lat','lon','time_idx','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']].copy()
    lag['time_idx'] += lp
    lag['k'] = lag['lat'].astype(str)+'_'+lag['lon'].astype(str)+'_'+lag['time_idx'].astype(str)
    lag = lag[['k','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']].rename(columns={
        'TWS_t':f'TWS_l{lp}','SOIL_MOISTURE_t':f'SOIL_l{lp}','SPEI_01_t':f'SPEI_l{lp}'})
    train['k'] = train['lat'].astype(str)+'_'+train['lon'].astype(str)+'_'+(train['time_idx']-lp).astype(str)
    test['k'] = test['lat'].astype(str)+'_'+test['lon'].astype(str)+'_'+(test['time_idx']-lp).astype(str)
    train = train.merge(lag, on='k', how='left')
    test = test.merge(lag, on='k', how='left')

# Fallback
last_t = train[train['time_idx']==train['time_idx'].max()][['lat','lon','TWS_t','SOIL_MOISTURE_t','SPEI_01_t']]
last_t.columns = ['lat','lon','TWS_lf','SOIL_lf','SPEI_lf']
test = test.merge(last_t, on=['lat','lon'], how='left')
for lp in [1,2,3,6]:
    test[f'TWS_l{lp}'] = test[f'TWS_l{lp}'].fillna(test['TWS_lf'])
    test[f'SOIL_l{lp}'] = test[f'SOIL_l{lp}'].fillna(test['SOIL_lf'])
    test[f'SPEI_l{lp}'] = test[f'SPEI_l{lp}'].fillna(test['SPEI_lf'])

before = len(train)
train = train.dropna(subset=['TWS_l1']).reset_index(drop=True)
print(f"Train: {len(train)}")

# FEATURES
for df in [train, test]:
    df['abs_lat'] = df['lat'].abs()
    df['spei_mean'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].mean(axis=1)
    df['spei_std'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].std(axis=1)
    df['tws_soil'] = df['TWS_t']*df['SOIL_MOISTURE_t']
    df['tws_spei01'] = df['TWS_t']*df['SPEI_01_t']
    df['tws_l1d'] = df['TWS_t']-df['TWS_l1']
    df['tws_l3d'] = df['TWS_t']-df['TWS_l3']
    df['soil_l1d'] = df['SOIL_MOISTURE_t']-df['SOIL_l1']
    df['trend_13'] = df['TWS_l1']-df['TWS_l3']
    df['trend_16'] = df['TWS_l1']-df['TWS_l6']
    df['lag_avg'] = df[['TWS_l1','TWS_l2','TWS_l3']].mean(axis=1)

features = [
    'TWS_t','SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t',
    'month_sin','month_cos','lat','lon','abs_lat','month',
    'spei_mean','spei_std','tws_soil','tws_spei01',
    'TWS_l1','SOIL_l1','SPEI_l1','TWS_l2','TWS_l3','TWS_l6',
    'tws_l1d','tws_l3d','soil_l1d','trend_13','trend_16','lag_avg',
]

X = train[features].values
y = train['target'].values
Xt = test[features].values
print(f"Features: {len(features)}")

params = {
    'objective':'regression','metric':'rmse','boosting_type':'gbdt',
    'num_leaves':255,'learning_rate':0.05,'feature_fraction':0.8,
    'bagging_fraction':0.8,'bagging_freq':5,'n_estimators':800,
    'verbose':-1,'seed':SEED,'n_jobs':-1,'min_child_samples':50,
    'reg_alpha':0.1,'reg_lambda':0.1,
}

print("\nTraining 3-Fold CV...")
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
test_preds = np.zeros(len(Xt))
scores = []

for fold,(tri,vai) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/3...")
    m = lgb.LGBMRegressor(**params)
    m.fit(X[tri],y[tri],eval_set=[(X[vai],y[vai])],
          callbacks=[lgb.early_stopping(50),lgb.log_evaluation(200)])
    vp = m.predict(X[vai])
    test_preds += m.predict(Xt)/3
    rmse = np.sqrt(mean_squared_error(y[vai],vp))
    scores.append(rmse)
    print(f"  RMSE: {rmse:.6f}")

print(f"\nOOF RMSE (on unmasked train): {np.mean(scores):.6f}")

# POST-HOC FIX: For masked rows, blend with lag-based prediction
# When TWS_t is NaN, the model output is based on NaN handling
# Better: use lag-based prediction for masked rows
print("\nPost-hoc fix for masked rows...")
mask = test['TWS_t_masked'].values
# Lag-based prediction: TWS_t+1 ≈ TWS_lag1 (persistence) + some adjustment
# Actually target ≈ TWS_t (persistence). For masked rows, use lag avg
lag_pred = test['TWS_l1'].values  # best available proxy
# Blend: for masked rows, use lag_pred; for unmasked, use model
final_preds = np.where(mask, lag_pred, test_preds)

print(f"Model-only RMSE would be for all: need to check")
print(f"Blended predictions for {mask.sum()} masked rows")

sub = sample_sub.copy()
sub['Target'] = final_preds
sub.to_csv('submission_v6.csv', index=False)
print(f"\nSaved submission_v6.csv")

# Also save a pure model version
sub2 = sample_sub.copy()
sub2['Target'] = test_preds
sub2.to_csv('submission_v6_nofix.csv', index=False)
print(f"Saved submission_v6_nofix.csv (no fix)")
print(f"Time: {time.time()-t0:.0f}s")
