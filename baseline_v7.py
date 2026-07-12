"""
v7 FAST: Corrected approach - no lat/lon, time-based split, fewer iterations
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
fit_times = unique_times[:split_idx]
val_times = unique_times[split_idx:]

fit_df = train[train['time'].isin(fit_times)]
val_df = train[train['time'].isin(val_times)]

X_fit = fit_df[features].values
y_fit = fit_df['target'].values
X_val = val_df[features].values
y_val = val_df['target'].values
print(f"Fit: {len(fit_df)}, Val: {len(val_df)}")

persist_rmse = np.sqrt(mean_squared_error(y_val, val_df['TWS_t'].values))
print(f"Persistence: {persist_rmse:.6f}")

# HGB - reduced iterations
print("\nTraining HGB...")
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('gbr', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.05, max_iter=300,
        max_depth=8, min_samples_leaf=50, l2_regularization=1.0,
        random_state=SEED
    ))
])
model.fit(X_fit, y_fit)
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"HGB Val RMSE: {rmse:.6f} (persist: {persist_rmse:.6f})")

# FINAL on all data
print("\nTraining final on all data...")
X_all = train[features].values
y_all = train['target'].values
X_test = test[features].values

model.fit(X_all, y_all)
final_pred = model.predict(X_test)

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv('submission_v7.csv', index=False)
print(f"Saved submission_v7.csv")
print(f"Pred: mean={final_pred.mean():.4f}, std={final_pred.std():.4f}")
print(f"Time: {time.time()-t0:.0f}s")
