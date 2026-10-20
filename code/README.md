# Simulation code for `main_ver1.tex`

Python 3 with `numpy`, `scipy` and `matplotlib`.

| File | Purpose |
|---|---|
| `tweedie_gmm.py` | DGP, Tweedie GLM (stage 1), weighted moments and nonparametric estimator with stage-1-corrected covariance, minimum-distance/GMM fits of S1 (θ-AR(1)), S2 (log-AR(1)) and S3 (exchangeable), J and Wald tests, Theorem 1 and Proposition 2 premiums, exact MSE |
| `test_core.py` | Unit tests: Kalman recursion = Theorem 1, weight bounds, recency counterexample, simulated vs exact MSE |
| `sim.py` | Study A/B grid, sensitivity and null designs, cluster bootstrap |
| `make_tables.py` | Builds `../results/tables/*.tex`, `../figures/*.pdf`, `../results/summary.json` and inlines the tables into `../main_ver1.tex` |

Reproduce (about 40 minutes on 4 cores):

```
python3 test_core.py
python3 sim.py all 500                 # main grid + sensitivity cells -> ../results/*.npz
python3 sim.py exact                   # exact population MSEs (2e6 policies per cell)
python3 sim.py boot 200 grid_v0.1_r0.3,grid_v0.5_r0.7,grid_v2_r0.9
python3 make_tables.py
cd .. && latexmk -pdf main_ver1.tex
```
