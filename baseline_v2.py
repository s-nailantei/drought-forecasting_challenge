"""
Baseline v2 FAST: No groupby, just numpy shift trick
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
print(f"Train: {train.shape}, Test: {test.shape}")

train['time'] = pd.to_datetime(train['time'])
test['time'] = pd.to_datetime(test['time'])
train['month'] = train['time'].dt.month
test['month'] = test['time'].dt.month

# Check unique times
print(f"Train times: {train['time'].nunique()} unique, range: {train['time'].min()} to {train['time'].max()}")
print(f"Test times: {test['time'].nunique()} unique, range: {test['time'].min()} to {test['time'].max()}")

# Quick engineered features (NO groupby)
for df in [train, test]:
    df['abs_lat'] = df['lat'].abs()
    df['spei_mean'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].mean(axis=1)
    df['spei_std'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].std(axis=1)
    df['spei_range'] = df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].max(axis=1) - df[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].min(axis=1)
    df['tws_soil'] = df['TWS_t'] * df['SOIL_MOISTURE_t']
    df['tws_spei01'] = df['TWS_t'] * df['SPEI_01_t']
    df['soil_spei01'] = df['SOIL_MOISTURE_t'] * df['SPEI_01_t']

features = [
    'TWS_t', 'SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t',
    'month_sin', 'month_cos', 'lat', 'lon', 'abs_lat', 'month',
    'spei_mean', 'spei_std', 'spei_range', 'tws_soil', 'tws_spei01', 'soil_spei01',
]

X_train = train[features].values
y_train = train['target'].values
X_test = test[features].values
print(f"Features: {len(features)}, Training: {len(X_train)}")

lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 255, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'n_estimators': 1000,
    'verbose': -1, 'seed': SEED, 'n_jobs': -1,
    'min_child_samples': 50,
}

print("\nTraining 3-Fold CV...")
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
test_preds = np.zeros(len(X_test))
rmse_scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}/3...")
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train[tr_idx], y_train[tr_idx],
        eval_set=[(X_train[val_idx], y_train[val_idx])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)]
    )
    val_pred = model.predict(X_train[val_idx])
    test_preds += model.predict(X_test) / 3
    rmse = np.sqrt(mean_squared_error(y_train[val_idx], val_pred))
    rmse_scores.append(rmse)
    print(f"  Fold {fold+1} RMSE: {rmse:.6f}")

persist_rmse = np.sqrt(mean_squared_error(y_train, train['TWS_t'].values))
print(f"\nMean OOF RMSE: {np.mean(rmse_scores):.6f}")
print(f"Persistence baseline: {persist_rmse:.6f}")
print(f"v1 was 0.4571")

submission = sample_sub.copy()
submission['Target'] = test_preds
submission.to_csv('submission_v2.csv', index=False)
print(f"\nSaved submission_v2.csv")
print(f"Time: {time.time()-t0:.0f}s")
