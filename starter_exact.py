"""
EXACT starter notebook - reproduce 0.6595 benchmark
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import time

t0 = time.time()

# Load exactly as starter does
train_features = pd.read_csv('Train.csv', parse_dates=["time"])
test_features = pd.read_csv('Test.csv', parse_dates=["time"])
sample_submission_df = pd.read_csv('SampleSubmission.csv')

# Normalize IDs
if "sample_id" in train_features.columns and "ID" not in train_features.columns:
    train_features = train_features.rename(columns={"sample_id": "ID"})
if "sample_id" in test_features.columns and "ID" not in test_features.columns:
    test_features = test_features.rename(columns={"sample_id": "ID"})

print("train:", train_features.shape)
print("test:", test_features.shape)
print("Train target col:", "target" in train_features.columns, "Target" in train_features.columns)

# Starter: rename target to Target
train_features = train_features.rename(columns={"target": "Target"})

# EXACT starter features
available_common = sorted(
    (set(train_features.columns) & set(test_features.columns)) - {"ID", "time", "lat", "lon", "TWS_t_masked"}
)
print("Available:", available_common)

BASE_FEATURES = ["TWS_t", "month_sin", "month_cos"]
OPTIONAL_FEATURES = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
feature_names = BASE_FEATURES + [f for f in OPTIONAL_FEATURES if f in available_common]
print("Features:", feature_names)

# EXACT starter build_xy
def build_xy(df, feature_names, use_latlon=False):
    X = df[feature_names].to_numpy(dtype=np.float32)
    if use_latlon:
        latlon = df[["lat", "lon"]].to_numpy(dtype=np.float32)
        X = np.column_stack([X, latlon])
    y = df["Target"].to_numpy(dtype=np.float32) if "Target" in df.columns else None
    tws_t = df["TWS_t"].to_numpy(dtype=np.float32)
    return X, y, tws_t

# EXACT starter model
def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gbr", HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=300,
            max_depth=8, min_samples_leaf=50, l2_regularization=1.0,
            random_state=42
        ))
    ])

# TIME-BASED SPLIT (exact starter)
unique_times = np.sort(train_features["time"].unique())
split_idx = int(len(unique_times) * 0.8)
fit_times = unique_times[:split_idx]
val_times = unique_times[split_idx:]

fit_df = train_features[train_features["time"].isin(fit_times)].copy().reset_index(drop=True)
val_df = train_features[train_features["time"].isin(val_times)].copy().reset_index(drop=True)

print(f"Fit: {len(fit_df)}, Val: {len(val_df)}")

# Train on fit set
X_fit, y_fit, _ = build_xy(fit_df, feature_names)
X_val, y_val, tws_val = build_xy(val_df, feature_names)

model = make_model()
model.fit(X_fit, y_fit)

y_val_pred = model.predict(X_val)
rmse_model = np.sqrt(mean_squared_error(y_val, y_val_pred))
rmse_persist = np.sqrt(mean_squared_error(y_val, tws_val))
print(f"Val Model RMSE: {rmse_model:.6f}")
print(f"Val Persist RMSE: {rmse_persist:.6f}")

# FINAL: retrain on ALL data, predict test
print("\nFinal model on all training data...")
X_train_full, y_train_full, _ = build_xy(train_features, feature_names)
final_model = make_model()
final_model.fit(X_train_full, y_train_full)

# EXACT starter: merge sample_sub with test, then predict
test_submission_df = sample_submission_df.merge(
    test_features, on="ID", how="left", validate="one_to_one"
)
print(f"Test merged: {test_submission_df.shape}")
print(f"Test TWS_t NaN: {test_submission_df['TWS_t'].isna().sum()}/{len(test_submission_df)}")

X_test_submit, _, _ = build_xy(test_submission_df, feature_names)
print(f"X_test_submit NaN count: {np.isnan(X_test_submit).sum()}")

y_test_pred = final_model.predict(X_test_submit)

submission_df = sample_submission_df.copy()
submission_df["Target"] = y_test_pred.astype(np.float32)
submission_df.columns = ["ID", "Target"]
submission_df.to_csv('submission_starter.csv', index=False)

print(f"\nPred stats: mean={y_test_pred.mean():.4f}, std={y_test_pred.std():.4f}")
print(f"Saved submission_starter.csv")
print(f"Time: {time.time()-t0:.0f}s")
