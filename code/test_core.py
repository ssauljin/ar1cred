"""Unit tests: Theorem 1 vs Proposition 2 recursion, weight bounds, exact MSE."""
import numpy as np
import tweedie_gmm as tg

rng = np.random.default_rng(1)

# 1) Kalman recursion reproduces Theorem 1 for gamma(k) = v rho^k, incl. missing years
I, T = 500, 6
q = np.exp(rng.normal(0, 0.7, (I, T)))
mask = rng.random((I, T)) > 0.2
v, rho, a = 0.7, 0.6, 2.0
gam = v * rho ** np.arange(T + 1)
w_direct = tg.bl_weights(gam, a, q, mask)
w_kal, K = tg.kalman_weights(v, rho, a, q, mask)
print('max |w_direct - w_kalman| =', np.abs(w_direct - w_kal).max())
assert np.allclose(w_direct, w_kal, atol=1e-7)
print('min weight', w_kal[mask].min(), ' max weight sum', w_kal.sum(1).max(), '< rho =', rho)
assert (w_kal[mask] > 0).all() and (w_kal.sum(1) < rho).all()

# 2) exchangeable limit
w_ex = tg.exch_weights(v, a, q, mask)
w_ex2 = tg.bl_weights(np.full(T + 1, v), a, q, mask)
assert np.allclose(w_ex, w_ex2, atol=1e-8)

# 3) recency: constant mu -> increasing weights; counterexample with varying mu
qc = np.ones((1, T)) * 1.3
wc, _ = tg.kalman_weights(v, rho, a, qc, np.ones((1, T), bool))
print('constant-mu weights', np.round(wc, 4))
assert (np.diff(wc[0]) > 0).all()
qv = np.array([[0.01, 25.0]])
wv, _ = tg.kalman_weights(0.5, 0.8, 1.0, qv, np.ones((1, 2), bool))
print('counterexample weights (d_1 small, d_2 large):', np.round(wv, 4))

# 4) simulated MSE of oracle-parameter predictor matches the exact formula
v0, r1, phi, p = 0.5, 0.7, 10.0, 1.5
s2w, rhow = tg.logar1_params(v0, r1)
gam = tg.gamma_logar1(np.arange(6), s2w, rhow)
a0 = phi * np.exp(s2w * p * (p - 1) / 2)
sims, exs = [], []
for r in range(40):
    d = tg.simulate_panel(rng, I=20000, T=5, phi=phi, p=p, s2w=s2w, rhow=rhow)
    mu = d['mu']; q = mu[:, :5] ** (p - 2); e = d['Z'][:, :5] / mu[:, :5] - 1
    w = tg.bl_weights(gam, a0, q, d['mask'])
    P = tg.premium(mu[:, 5], w, e)
    sims.append(np.mean((d['Z'][:, 5] - P) ** 2))
    exs.append(tg.exact_mse(gam, a0, q, mu[:, 5] ** (p - 2), mu[:, 5], w, d['mask']).mean())
sims, exs = np.array(sims), np.array(exs)
print('simulated MSE %.3f (MC se %.3f)  exact %.3f' % (sims.mean(), sims.std() / np.sqrt(len(sims)), exs.mean()))
