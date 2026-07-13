"""
v9: Lag features + masking + no lat/lon features
THE definitive approach to close the gap to 0.655
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import warnings, time
warnings.filterwarnings('ignore')

SEED = 42
t0 = time.time()

print("Loading...")
train = pd.read_csv('Train.csv', parse_dates=["time"])
test = pd.read_csv('Test.csv', parse_dates=["time"])
sample_sub = pd.read_csv('SampleSubmission.csv')

if "sample_id" in train.columns:
    train = train.rename(columns={"sample_id": "ID", "target": "Target"})
if "sample_id" in test.columns:
    test = test.rename(columns={"sample_id": "ID"})

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Test masked: {test['TWS_t_masked'].sum()}/{len(test)}")

# ============================================
# LAG FEATURES (key innovation)
# Use location grouping to create lags, but DON'T use lat/lon as model features
# ============================================
print("Building lag features...")

# Create location key for grouping (not a model feature)
train['loc'] = train['lat'].astype(str) + '_' + train['lon'].astype(str)
test['loc'] = test['lat'].astype(str) + '_' + test['lon'].astype(str)

# Create time index
all_times = sorted(train['time'].unique())
time_idx = {t: i for i, t in enumerate(all_times)}
train['time_idx'] = train['time'].map(time_idx)
max_idx = max(time_idx.values())
test['time_idx'] = test['time'].apply(lambda t: time_idx.get(t, max_idx + 1))

# Lag lookup: for each (loc, time_idx), get TWS_t from time_idx-1
print("Creating lag lookups...")
lag_cols = ['TWS_t', 'SOIL_MOISTURE_t', 'SPEI_01_t']

for lag in [1, 2, 3, 6]:
    lag_data = train[['loc', 'time_idx'] + lag_cols].copy()
    lag_data['time_idx'] = lag_data['time_idx'] + lag
    lag_data = lag_data.rename(columns={c: f'{c}_lag{lag}' for c in lag_cols})
    
    key_new = f'_lag_key_{lag}'
    train[key_new] = train['loc'] + '_' + (train['time_idx'] - lag).astype(str)
    test[key_new] = test['loc'] + '_' + (test['time_idx'] - lag).astype(str)
    lag_data['_lag_key'] = lag_data['loc'] + '_' + lag_data['time_idx'].astype(str)
    
    lag_data = lag_data.drop(columns=['loc', 'time_idx']).drop_duplicates(subset=['_lag_key'])
    
    train = train.merge(lag_data, left_on=key_new, right_on='_lag_key', how='left')
    test = test.merge(lag_data, left_on=key_new, right_on='_lag_key', how='left')
    train = train.drop(columns=[key_new, '_lag_key'], errors='ignore')
    test = test.drop(columns=[key_new, '_lag_key'], errors='ignore')

# Fallback for test NaN lags: use the last training period
last_time = train['time_idx'].max()
last_vals = train[train['time_idx'] == last_time][['loc'] + [f'{c}_lag1' for c in lag_cols]]
last_vals.columns = ['loc'] + [f'{c}_last' for c in lag_cols]
test = test.merge(last_vals, on='loc', how='left')
for c in lag_cols:
    for lag in [1, 2, 3, 6]:
        col = f'{c}_lag{lag}'
        last_col = f'{c}_last'
        if col in test.columns and last_col in test.columns:
            test[col] = test[col].fillna(test[last_col])

# Drop rows with NaN lag1 in training (earliest time steps)
before = len(train)
train = train.dropna(subset=['TWS_t_lag1']).reset_index(drop=True)
print(f"Train: {len(train)} (dropped {before - len(train)})")

# ============================================
# FEATURES: NO lat/lon!
# ============================================
print("Feature engineering...")
features = [
    'TWS_t', 'month_sin', 'month_cos',
    'SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t',
    'TWS_t_lag1', 'TWS_t_lag2', 'TWS_t_lag3', 'TWS_t_lag6',
    'SOIL_MOISTURE_t_lag1', 'SPEI_01_t_lag1',
]

# Add interaction features (safe: no NaN issues for lag columns)
for df in [train, test]:
    df['tws_lag_diff'] = df['TWS_t'] - df['TWS_t_lag1']
    df['tws_lag_avg'] = df[['TWS_t_lag1', 'TWS_t_lag2', 'TWS_t_lag3']].mean(axis=1)
    df['tws_lag_trend'] = df['TWS_t_lag1'] - df['TWS_t_lag3']

features += ['tws_lag_diff', 'tws_lag_avg', 'tws_lag_trend']
print(f"Features ({len(features)}): {features}")

# ============================================
# TIME-BASED SPLIT
# ============================================
unique_times = np.sort(train['time'].unique())
split_idx = int(len(unique_times) * 0.8)
fit_df = train[train['time'].isin(unique_times[:split_idx])]
val_df = train[train['time'].isin(unique_times[split_idx:])]

X_fit = fit_df[features].values
y_fit = fit_df['Target'].values
X_val = val_df[features].values
y_val = val_df['Target'].values
X_test = test[features].values

persist_val = np.sqrt(mean_squared_error(y_val, val_df['TWS_t'].values))
print(f"\nVal persistence: {persist_val:.6f}")

# ============================================
# TRAIN WITH MASKING (67% TWS_t NaN)
# ============================================
np.random.seed(SEED)
mask_frac = test['TWS_t_masked'].mean()

# Mask TWS_t in training
X_fit_masked = X_fit.copy()
n_mask = int(len(X_fit_masked) * mask_frac)
mask_idx = np.random.choice(len(X_fit_masked), size=n_mask, replace=False)
X_fit_masked[mask_idx, 0] = np.nan  # TWS_t is column 0

print(f"Training with {n_mask} masked rows ({mask_frac*100:.0f}%)...")

# HGB with masking
hgb_masked = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=500,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))
])
hgb_masked.fit(X_fit_masked, y_fit)
hgb_val = np.sqrt(mean_squared_error(y_val, hgb_masked.predict(X_val)))
print(f"Masked HGB Val: {hgb_val:.6f}")

# HGB without masking (baseline)
hgb_std = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=500,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))
])
hgb_std.fit(X_fit, y_fit)
std_val = np.sqrt(mean_squared_error(y_val, hgb_std.predict(X_val)))
print(f"Standard HGB Val: {std_val:.6f}")

# ============================================
# FINAL: retrain on full data with masking
# ============================================
print("\n=== Final model ===")
X_all = train[features].values
y_all = train['Target'].values

# Apply masking to full training data
X_all_masked = X_all.copy()
mask_all = np.random.choice(len(X_all_masked), size=int(len(X_all_masked) * mask_frac), replace=False)
X_all_masked[mask_all, 0] = np.nan

final_model = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=500,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED))
])
final_model.fit(X_all_masked, y_all)

pred = final_model.predict(X_test)

sub = sample_sub.copy()
sub['Target'] = pred
sub.to_csv('submission_v9.csv', index=False)

print(f"Pred: mean={pred.mean():.4f}, std={pred.std():.4f}")
print(f"Train target: mean={y_all.mean():.4f}, std={y_all.std():.4f}")
print(f"\nVal comparison:")
print(f"  With lag features + masking: {hgb_val:.6f}")
print(f"  Without lag features: std={std_val:.6f}")
print(f"  Persist baseline: {persist_val:.6f}")
print(f"\nSaved submission_v9.csv")
print(f"Time: {time.time()-t0:.0f}s")
