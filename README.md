# A Step Ahead of Drought: Forecasting Global Water Storage Challenge

Zindi competition: Predict one-month ahead Total Water Storage (TWS) globally.

## Goal
Win the competition (1st place - €1,000)

## Data
- Train: 2,154,021 rows, 138 time steps (2002-05 to 2015-08)
- Test: 280,961 rows, 18 time steps (2015-09 to 2018-12)
- 15,715 unique locations
- Features: TWS_t, SPEI_01/03/06/12, SOIL_MOISTURE_t, month_sin, month_cos
- Target: TWS_t+1 (one-month ahead prediction)

## Evaluation
- Leaderboard: RMSE (50%)
- AI Trustworthiness (30%)
- Innovation & Practicality (20%)

## Model Progress

| Version | OOF RMSE | Key Changes |
|---------|----------|-------------|
| v1 | 0.4571 | Baseline LightGBM, 18 features |
| v2 | 0.4363 | + SPEI interactions, TWS*SOIL, TWS*SPEI |
| v3 | 0.4126 | + Lag features (lag1, lag3) via time-index merge |
| **v4** | **0.3582** | + Multi-period lags (1,2,3,6), trends, 1200 trees |
| v5 | 0.3629 | + Spatial regional features, but fewer trees |

**Best: v4 (OOF RMSE 0.3582)** — 22% improvement over persistence baseline (0.5724)

## Key Features (by importance)
1. lon, lat (spatial position)
2. TWS_t_lag6, TWS_t_lag2, TWS_t_lag3 (temporal lags)
3. tws_trend_1_3, tws_lag3_diff (trend features)
4. month_sin, month_cos (seasonal encoding)

## Next Steps
- Try higher learning rate + more trees (still not early stopping)
- Add target encoding for spatial features
- Try CatBoost / XGBoost ensemble
- Time-based CV instead of random CV
- Add Copernicus external data

## Deadline
Sep 13, 2026
