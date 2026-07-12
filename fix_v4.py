"""
v6: Fix v4 submission for masked test rows
Simple approach: load v4 predictions, replace masked rows with lag-based estimate
"""
import pandas as pd
import numpy as np
import time

t0 = time.time()

test = pd.read_csv('Test.csv')
v4_sub = pd.read_csv('submission_v4.csv')

print(f"Test: {test.shape}")
print(f"v4 submission: {v4_sub.shape}")
print(f"Masked rows: {test['TWS_t_masked'].sum()}/{len(test)}")

# For masked rows, TWS_t is NaN - our model produced bad predictions
# Use persistence: TWS_t+1 ≈ TWS_lag1 (or TWS_t if available)
# Since TWS_t is NaN for masked rows, we need lag features
# But we don't have lag features in the test file directly

# Strategy: For masked rows, the best simple predictor is the 
# monthly mean of TWS_t at that location from training data
# Or: use SOIL_MOISTURE and SPEI to predict

# Actually the simplest: for masked rows, use 0 (global mean of target)
# Since target is normalized with mean ~0.11
# Better: use the non-masked row predictions' mean for same time/location

# Let's check: what does the model predict for masked vs non-masked?
test_with_pred = test.copy()
test_with_pred['pred'] = v4_sub['Target'].values

print("\nPrediction stats by mask status:")
print("Non-masked predictions:")
print(test_with_pred[~test_with_pred['TWS_t_masked']]['pred'].describe())
print("\nMasked predictions (likely garbage):")
print(test_with_pred[test_with_pred['TWS_t_masked']]['pred'].describe())

# The masked predictions have different distribution - they're garbage
# Fix: use TWS_lag1 as prediction for masked rows
# But we need lag data... let's build it quickly

test['time'] = pd.to_datetime(test['time'])
train = pd.read_csv('Train.csv', usecols=['lat','lon','time','TWS_t'])
train['time'] = pd.to_datetime(train['time'])

# Get last known TWS_t per location (from training)
last_tws = train.sort_values('time').groupby(['lat','lon']).last().reset_index()
last_tws = last_tws[['lat','lon','TWS_t']].rename(columns={'TWS_t':'TWS_last_known'})

test_merged = test.merge(last_tws, on=['lat','lon'], how='left')

# Fix predictions for masked rows
preds = v4_sub['Target'].values.copy()
mask = test['TWS_t_masked'].values

# For masked rows, use last known TWS as proxy for TWS_t+1
# (TWS changes slowly, so TWS_last_known ≈ TWS_t ≈ TWS_t+1)
fixed_preds = np.where(mask, test_merged['TWS_last_known'].fillna(0), preds)

sub = v4_sub.copy()
sub['Target'] = fixed_preds
sub.to_csv('submission_v6.csv', index=False)

print(f"\nBefore fix: mean={preds.mean():.4f}, std={preds.std():.4f}")
print(f"After fix: mean={fixed_preds.mean():.4f}, std={fixed_preds.std():.4f}")
print(f"Train target mean: 0.1125, std: 0.912")

sub_nofix = v4_sub.copy()
sub_nofix.to_csv('submission_v6_nofix.csv', index=False)

print(f"\nSaved submission_v6.csv (with fix)")
print(f"Saved submission_v6_nofix.csv (v4 original)")
print(f"Time: {time.time()-t0:.0f}s")
