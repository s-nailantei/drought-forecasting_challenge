"""
Baseline v1: LightGBM for TWS Forecasting - SPEED OPTIMIZED
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings, time
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 3
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
train['year'] = train['time'].dt.year

print("Feature engineering...")
train['abs_lat'] = train['lat'].abs()
test['abs_lat'] = test['lat'].abs()

train['spei_mean'] = train[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].mean(axis=1)
test['spei_mean'] = test[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].mean(axis=1)

train['spei_std'] = train[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].std(axis=1)
test['spei_std'] = test[['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']].std(axis=1)

train['spei_diff_01_12'] = train['SPEI_01_t'] - train['SPEI_12_t']
test['spei_diff_01_12'] = test['SPEI_01_t'] - test['SPEI_12_t']

train['tws_soil'] = train['TWS_t'] * train['SOIL_MOISTURE_t']
test['tws_soil'] = test['TWS_t'] * test['SOIL_MOISTURE_t']

train['tws_spei01'] = train['TWS_t'] * train['SPEI_01_t']
test['tws_spei01'] = test['TWS_t'] * test['SPEI_01_t']

features = [
    'TWS_t', 'SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t',
    'month_sin', 'month_cos', 'lat', 'lon', 'abs_lat', 'month',
    'spei_mean', 'spei_std', 'spei_diff_01_12', 'tws_soil', 'tws_spei01'
]

X_train = train[features].values
y_train = train['target'].values
X_test = test[features].values

print(f"Features: {len(features)}, Samples: {len(X_train)}")

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_estimators': 500,
    'verbose': -1,
    'seed': SEED,
    'n_jobs': -1,
    'min_child_samples': 100,
}

print("\nTraining with 3-Fold CV...")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
test_preds = np.zeros(len(X_test))
rmse_scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}/{N_FOLDS}...")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    val_pred = model.predict(X_val)
    test_preds += model.predict(X_test) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    rmse_scores.append(rmse)
    print(f"  Fold {fold+1} RMSE: {rmse:.6f}")

persist_rmse = np.sqrt(mean_squared_error(y_train, train['TWS_t'].values))
print(f"\nMean OOF RMSE: {np.mean(rmse_scores):.6f} (+/- {np.std(rmse_scores):.6f})")
print(f"Persistence baseline RMSE: {persist_rmse:.6f}")

submission = sample_sub.copy()
submission['Target'] = test_preds
submission.to_csv('submission_v1.csv', index=False)
print(f"\nSaved: submission_v1.csv")
print(f"Time: {time.time()-t0:.0f}s")
print(submission['Target'].describe())
