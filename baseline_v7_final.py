"""
v7 final: HGB + LGB ensemble, no lat/lon
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
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

unique_times = np.sort(train['time'].unique())
split_idx = int(len(unique_times) * 0.8)
fit_df = train[train['time'].isin(unique_times[:split_idx])]
val_df = train[train['time'].isin(unique_times[split_idx:])]

X_fit = fit_df[features].values
y_fit = fit_df['target'].values
X_val = val_df[features].values
y_val = val_df['target'].values
X_test = test[features].values

# LGB (fast, already known: 0.5982)
print("LGB...")
lgb_m = lgb.LGBMRegressor(
    objective='regression', metric='rmse', boosting_type='gbdt',
    num_leaves=255, learning_rate=0.05, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, n_estimators=500,
    verbose=-1, seed=SEED, n_jobs=-1, min_child_samples=50)
lgb_m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
rmse_lgb = np.sqrt(mean_squared_error(y_val, lgb_m.predict(X_val)))
print(f"LGB Val: {rmse_lgb:.6f}")

# HGB (reduced to fit in time)
print("HGB...")
hgb = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=300,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))
])
hgb.fit(X_fit, y_fit)
rmse_hgb = np.sqrt(mean_squared_error(y_val, hgb.predict(X_val)))
print(f"HGB Val: {rmse_hgb:.6f}")

# FINAL
print("Final...")
lgb_final = lgb.LGBMRegressor(
    objective='regression', metric='rmse', boosting_type='gbdt',
    num_leaves=255, learning_rate=0.05, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, n_estimators=500,
    verbose=-1, seed=SEED, n_jobs=-1, min_child_samples=50)
lgb_final.fit(train[features].values, train['target'].values)
pred_lgb = lgb_final.predict(X_test)

hgb.fit(train[features].values, train['target'].values)
pred_hgb = hgb.predict(X_test)

pred_ens = (pred_lgb + pred_hgb) / 2

sub = sample_sub.copy()
sub['Target'] = pred_lgb
sub.to_csv('submission_v7_lgb.csv', index=False)

sub2 = sample_sub.copy()
sub2['Target'] = pred_hgb
sub2.to_csv('submission_v7_hgb.csv', index=False)

sub3 = sample_sub.copy()
sub3['Target'] = pred_ens
sub3.to_csv('submission_v7_ensemble.csv', index=False)

print(f"\nLGB Val: {rmse_lgb:.6f}, HGB Val: {rmse_hgb:.6f}")
print(f"Saved: v7_lgb, v7_hgb, v7_ensemble")
print(f"Time: {time.time()-t0:.0f}s")
