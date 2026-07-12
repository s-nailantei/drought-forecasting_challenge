"""
v7b: LightGBM only, no lat/lon, time-based split
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings, time
warnings.filterwarnings('ignore')

SEED = 42
t0 = time.time()

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sample_sub = pd.read_csv('SampleSubmission.csv')
if 'sample_id' in test.columns:
    test = test.rename(columns={'sample_id': 'ID'})

train['time'] = pd.to_datetime(train['time'])
test['time'] = pd.to_datetime(test['time'])

features = ['TWS_t', 'month_sin', 'month_cos', 'SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

# TIME-BASED SPLIT
unique_times = np.sort(train['time'].unique())
split_idx = int(len(unique_times) * 0.8)
fit_df = train[train['time'].isin(unique_times[:split_idx])]
val_df = train[train['time'].isin(unique_times[split_idx:])]

X_fit = fit_df[features].values
y_fit = fit_df['target'].values
X_val = val_df[features].values
y_val = val_df['target'].values

persist_rmse = np.sqrt(mean_squared_error(y_val, val_df['TWS_t'].values))
print(f"Persistence: {persist_rmse:.6f}")

# LGB with optimized params
print("Training LGB...")
lgb_m = lgb.LGBMRegressor(
    objective='regression', metric='rmse', boosting_type='gbdt',
    num_leaves=255, learning_rate=0.05, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, n_estimators=500,
    verbose=-1, seed=SEED, n_jobs=-1, min_child_samples=50)
lgb_m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
rmse_lgb = np.sqrt(mean_squared_error(y_val, lgb_m.predict(X_val)))
print(f"LGB Val: {rmse_lgb:.6f}")

# FINAL
print("Final training...")
lgb_final = lgb.LGBMRegressor(
    objective='regression', metric='rmse', boosting_type='gbdt',
    num_leaves=255, learning_rate=0.05, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, n_estimators=500,
    verbose=-1, seed=SEED, n_jobs=-1, min_child_samples=50)
lgb_final.fit(train[features].values, train['target'].values)

pred_lgb = lgb_final.predict(test[features].values)
sub = sample_sub.copy()
sub['Target'] = pred_lgb
sub.to_csv('submission_v7_lgb.csv', index=False)
print(f"Saved submission_v7_lgb.csv")
print(f"Time: {time.time()-t0:.0f}s")
