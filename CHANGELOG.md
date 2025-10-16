# Changelog

## v0.2
- Improved model (Ridge) vs v0.1 baseline (LinearRegression).
- Metrics:
  - RMSE (test): 53.7775
  - Delta vs v0.1: 53.8534 → 53.7775 (−0.0759)
- Rationale: L2 regularization (Ridge) reduces variance compared to OLS, yielding a modest RMSE gain.
 - Image: `ghcr.io/nm00001/assignment_3_mlops:v0.2`

## v0.1
- Baseline pipeline: `StandardScaler` + `LinearRegression`.
- Endpoints: `/health`, `/predict`.
- Metrics:
  - RMSE (test): 53.8534
 - Image: `ghcr.io/nm00001/assignment_3_mlops:v0.1`
