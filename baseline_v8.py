"""
v8: Train with simulated TWS_t masking so model handles NaN test rows
This is the ROOT FIX for the 0.60 -> 0.89 score gap
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

train = pd.read_csv('Train.csv', parse_dates=["time"])
test = pd.read_csv('Test.csv', parse_dates=["time"])
sample_sub = pd.read_csv('SampleSubmission.csv')

if "sample_id" in train.columns:
    train = train.rename(columns={"sample_id": "ID", "target": "Target"})
if "sample_id" in test.columns:
    test = test.rename(columns={"sample_id": "ID"})

features = ['TWS_t', 'month_sin', 'month_cos', 'SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']
print(f"Test masked: {test['TWS_t_masked'].sum()}/{len(test)} ({test['TWS_t_masked'].mean()*100:.0f}%)")

# TIME-BASED SPLIT
unique_times = np.sort(train["time"].unique())
split_idx = int(len(unique_times) * 0.8)
fit_df = train[train["time"].isin(unique_times[:split_idx])].copy()
val_df = train[train["time"].isin(unique_times[split_idx:])].copy()

X_fit = fit_df[features].values
y_fit = fit_df['Target'].values
X_val = val_df[features].values
y_val = val_df['Target'].values
X_test = test[features].values

persist_val = np.sqrt(mean_squared_error(y_val, val_df['TWS_t'].values))
print(f"Val persistence: {persist_val:.6f}")

# ============================================
# MODEL 1: Standard (no masking) - baseline
# ============================================
print("\n--- Standard HGB (no masking) ---")
std_model = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=300,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))
])
std_model.fit(X_fit, y_fit)
std_val = np.sqrt(mean_squared_error(y_val, std_model.predict(X_val)))
print(f"Val RMSE: {std_val:.6f}")

# ============================================
# MODEL 2: Train WITH masked TWS_t
# Key: set ~67% of TWS_t to NaN during training
# so model learns to predict without it
# ============================================
print("\n--- Masked HGB (67% TWS_t NaN) ---")
np.random.seed(SEED)

# Create masked version of fit data
X_fit_masked = X_fit.copy()
mask_frac = test['TWS_t_masked'].mean()  # ~67%
n_mask = int(len(X_fit_masked) * mask_frac)
mask_idx = np.random.choice(len(X_fit_masked), size=n_mask, replace=False)
X_fit_masked[mask_idx, 0] = np.nan  # TWS_t is column 0

print(f"Masked {n_mask}/{len(X_fit_masked)} rows ({mask_frac*100:.0f}%)")

masked_model = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=300,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))
])
masked_model.fit(X_fit_masked, y_fit)
masked_val = np.sqrt(mean_squared_error(y_val, masked_model.predict(X_val)))
print(f"Val RMSE (on unmasked val): {masked_val:.6f}")

# ============================================
# MODEL 3: LightGBM with masking
# ============================================
print("\n--- Masked LGB ---")
lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 255, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'n_estimators': 500,
    'verbose': -1, 'seed': SEED, 'n_jobs': -1, 'min_child_samples': 50,
}
lgb_masked = lgb.LGBMRegressor(**lgb_params)
lgb_masked.fit(X_fit_masked, y_fit, eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
lgb_masked_val = np.sqrt(mean_squared_error(y_val, lgb_masked.predict(X_val)))
print(f"Val RMSE (on unmasked val): {lgb_masked_val:.6f}")

# Also train standard LGB for comparison
print("\n--- Standard LGB ---")
lgb_std = lgb.LGBMRegressor(**lgb_params)
lgb_std.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
lgb_std_val = np.sqrt(mean_squared_error(y_val, lgb_std.predict(X_val)))
print(f"Val RMSE: {lgb_std_val:.6f}")

# ============================================
# FINAL: retrain masked models on full data
# ============================================
print("\n=== Final models (full training) ===")
X_all = train[features].values
y_all = train['Target'].values

# Masked HGB
X_all_masked = X_all.copy()
mask_idx_all = np.random.choice(len(X_all_masked), size=int(len(X_all_masked) * mask_frac), replace=False)
X_all_masked[mask_idx_all, 0] = np.nan

final_masked_hgb = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=300,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))
])
final_masked_hgb.fit(X_all_masked, y_all)

# Standard HGB (for comparison)
final_std_hgb = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=300,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))
])
final_std_hgb.fit(X_all, y_all)

# Masked LGB
final_masked_lgb = lgb.LGBMRegressor(**lgb_params)
final_masked_lgb.fit(X_all_masked, y_all)

# Standard LGB
final_std_lgb = lgb.LGBMRegressor(**lgb_params)
final_std_lgb.fit(X_all, y_all)

# Predictions
pred_std_hgb = final_std_hgb.predict(X_test)
pred_masked_hgb = final_masked_hgb.predict(X_test)
pred_std_lgb = final_std_lgb.predict(X_test)
pred_masked_lgb = final_masked_lgb.predict(X_test)
pred_ensemble = (pred_masked_hgb + pred_masked_lgb) / 2

# Save all
for name, pred in [
    ('v8_std_hgb', pred_std_hgb),
    ('v8_masked_hgb', pred_masked_hgb),
    ('v8_masked_lgb', pred_masked_lgb),
    ('v8_ensemble', pred_ensemble),
]:
    sub = sample_sub.copy()
    sub['Target'] = pred
    sub.to_csv(f'submission_{name}.csv', index=False)
    print(f"{name}: mean={pred.mean():.4f}, std={pred.std():.4f}")

print(f"\nVal comparison:")
print(f"  Standard HGB: {std_val:.6f}")
print(f"  Masked HGB:   {masked_val:.6f}")
print(f"  Standard LGB: {lgb_std_val:.6f}")
print(f"  Masked LGB:   {lgb_masked_val:.6f}")
print(f"Time: {time.time()-t0:.0f}s")
