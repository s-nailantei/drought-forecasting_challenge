"""
v9c: Single strong model + calibration
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

train = pd.read_csv('Train.csv', parse_dates=["time"])
test = pd.read_csv('Test.csv', parse_dates=["time"])
sample_sub = pd.read_csv('SampleSubmission.csv')
if "sample_id" in train.columns:
    train = train.rename(columns={"sample_id": "ID", "target": "Target"})
if "sample_id" in test.columns:
    test = test.rename(columns={"sample_id": "ID"})

# LAGS
train['loc'] = train['lat'].astype(str)+'_'+train['lon'].astype(str)
test['loc'] = test['lat'].astype(str)+'_'+test['lon'].astype(str)
all_times = sorted(train['time'].unique())
time_idx = {t:i for i,t in enumerate(all_times)}
train['ti'] = train['time'].map(time_idx)
mx = max(time_idx.values())
test['ti'] = test['time'].apply(lambda t: time_idx.get(t, mx+1))

lc = ['TWS_t','SOIL_MOISTURE_t','SPEI_01_t']
for lag in [1,2,3,6]:
    ld = train[['loc','ti']+lc].copy()
    ld['ti'] += lag
    ld = ld.rename(columns={c:f'{c}_l{lag}' for c in lc})
    ld['_k'] = ld['loc']+'_'+ld['ti'].astype(str)
    ld = ld.drop(columns=['loc','ti']).drop_duplicates(subset=['_k'])
    train['_k'] = train['loc']+'_'+(train['ti']-lag).astype(str)
    test['_k'] = test['loc']+'_'+(test['ti']-lag).astype(str)
    train = train.merge(ld, left_on='_k', right_on='_k', how='left')
    test = test.merge(ld, left_on='_k', right_on='_k', how='left')
    train.drop(columns=['_k'], inplace=True, errors='ignore')
    test.drop(columns=['_k'], inplace=True, errors='ignore')

lt = train['ti'].max()
lv = train[train['ti']==lt][['loc']+[f'{c}_l1' for c in lc]]
lv.columns = ['loc']+[f'{c}_last' for c in lc]
test = test.merge(lv, on='loc', how='left')
for c in lc:
    for lag in [1,2,3,6]:
        col=f'{c}_l{lag}'
        if col in test.columns:
            test[col]=test[col].fillna(test[f'{c}_last'])

train = train.dropna(subset=['TWS_t_l1']).reset_index(drop=True)

features = ['TWS_t','month_sin','month_cos','SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t',
            'TWS_t_l1','TWS_t_l2','TWS_t_l3','TWS_t_l6','SOIL_MOISTURE_t_l1','SPEI_01_t_l1']
for df in [train,test]:
    df['tld']=df['TWS_t']-df['TWS_t_l1']
    df['tla']=df[['TWS_t_l1','TWS_t_l2','TWS_t_l3']].mean(axis=1)
    df['tlt']=df['TWS_t_l1']-df['TWS_t_l3']
features+=['tld','tla','tlt']

X_all = train[features].values
y_all = train['Target'].values
X_test = test[features].values

mask_frac = test['TWS_t_masked'].mean()
np.random.seed(SEED)
X_masked = X_all.copy()
n_mask = int(len(X_masked)*mask_frac)
mi = np.random.choice(len(X_masked), size=n_mask, replace=False)
X_masked[mi, 0] = np.nan

print(f"Training masked HGB (800 iter)...")
model = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('m', HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=0.03, max_iter=800,
        max_depth=10, min_samples_leaf=30, l2_regularization=0.5, random_state=SEED))
])
model.fit(X_masked, y_all)
raw_pred = model.predict(X_test)

# CALIBRATION: match training target distribution
pred_cal = (raw_pred - raw_pred.mean()) / (raw_pred.std()+1e-8) * y_all.std() + y_all.mean()

sub = sample_sub.copy()
sub['Target'] = pred_cal
sub.to_csv('submission_v9c.csv', index=False)

print(f"Raw: mean={raw_pred.mean():.4f}, std={raw_pred.std():.4f}")
print(f"Cal: mean={pred_cal.mean():.4f}, std={pred_cal.std():.4f}")
print(f"Target: mean={y_all.mean():.4f}, std={y_all.std():.4f}")
print(f"Saved submission_v9c.csv, Time: {time.time()-t0:.0f}s")
