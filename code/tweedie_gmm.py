"""Tweedie evolutionary credibility with a second-order stationary latent process.

Core library: data-generating process, stage-1 Tweedie quasi-likelihood GLM,
stage-2 semi-parametric GMM / minimum-distance estimation of the latent
autocovariance gamma(k) and the nugget constant a = phi * E[theta^p],
specification and Wald tests, and the credibility premium of Theorem 1
(direct solve) and Proposition 2 (Kalman recursion).

Notation follows the manuscript:
    mu_it  = exp(x_it' beta)            GLM mean
    e_it   = Z_it / mu_it - 1           standardized residual
    q_it   = mu_it^(p-2)                so that Var(e_it | x) = gamma(0) + a q_it
    eta    = (gamma(0), a, gamma(1), ..., gamma(T-1))   nonparametric moment vector
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import chi2

# ----------------------------------------------------------------------------
# Data-generating process
# ----------------------------------------------------------------------------


def logar1_params(v, r1):
    """Map theta-scale (Var theta, lag-1 correlation) to log-scale (sigma_W^2, rho_W)."""
    s2w = np.log1p(v)
    rhow = np.log1p(v * r1) / s2w
    return s2w, rhow


def gamma_logar1(k, s2w, rhow):
    return np.expm1(s2w * rhow ** np.asarray(k, dtype=float))


def rtweedie(mu, phi, p, rng):
    """Compound Poisson-gamma draw with E Z = mu and Var Z = phi mu^p (1 < p < 2)."""
    lam = mu ** (2 - p) / (phi * (2 - p))
    shape = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu ** (p - 1)
    n = rng.poisson(lam)
    z = np.zeros_like(mu)
    pos = n > 0
    z[pos] = rng.gamma(n[pos] * shape, scale[pos])
    return z


def simulate_panel(rng, I=2000, T=5, beta=(2.5, 0.5, -0.5), phi=10.0, p=1.5,
                   s2w=0.5, rhow=0.7, sd_b=0.6, sd_u=0.8, miss=0.0):
    """Simulate a balanced (or randomly unbalanced) panel with T training periods
    and one hold-out period T+1.

    x_{1,it} = b_i + u_it (time-varying), x_{2,i} ~ Bernoulli(0.5) (fixed);
    W_it = log theta_it is a stationary Gaussian AR(1) with variance s2w,
    autocorrelation rhow and mean -s2w/2, so that E theta = 1.
    """
    beta = np.asarray(beta, float)
    Tp = T + 1
    x1 = rng.normal(0, sd_b, (I, 1)) + rng.normal(0, sd_u, (I, Tp))
    x2 = np.repeat(rng.binomial(1, 0.5, (I, 1)), Tp, axis=1).astype(float)
    X = np.stack([np.ones((I, Tp)), x1, x2], axis=2)
    mu = np.exp(X @ beta)
    W = np.empty((I, Tp))
    sd = np.sqrt(s2w)
    W[:, 0] = rng.normal(0, sd, I)
    inn = np.sqrt(max(1 - rhow ** 2, 0.0)) * sd
    for t in range(1, Tp):
        W[:, t] = rhow * W[:, t - 1] + inn * rng.normal(0, 1, I)
    theta = np.exp(W - s2w / 2)
    Z = rtweedie(mu * theta, phi, p, rng)
    mask = np.ones((I, T), bool)
    if miss > 0:
        mask = rng.random((I, T)) > miss
        # keep at least one observed training year per policy
        none = ~mask.any(axis=1)
        mask[none, T - 1] = True
    return dict(X=X, mu=mu, theta=theta, Z=Z, mask=mask, T=T)


# ----------------------------------------------------------------------------
# Stage 1: Tweedie quasi-likelihood GLM under working independence
# ----------------------------------------------------------------------------


def fit_glm(X, Z, mask, p, maxit=50, tol=1e-10):
    """Fisher scoring for the log-link Tweedie quasi-score sum x (Z - mu) mu^(1-p)."""
    Xo = X[mask]
    Zo = Z[mask]
    beta = np.zeros(Xo.shape[1])
    beta[0] = np.log(Zo.mean())
    for _ in range(maxit):
        mu = np.exp(Xo @ beta)
        score = Xo.T @ ((Zo - mu) * mu ** (1 - p))
        info = (Xo * (mu ** (2 - p))[:, None]).T @ Xo
        step = np.linalg.solve(info, score)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def glm_score_parts(X, Z, mask, beta, p):
    """Per-policy quasi-score s_i and the averaged expected Hessian S."""
    mu = np.exp(X @ beta)
    r = (Z - mu) * mu ** (1 - p) * mask
    s = np.einsum('itj,it->ij', X, r)
    wt = mu ** (2 - p) * mask
    S = -np.einsum('itj,itk,it->jk', X, X, wt) / X.shape[0]
    return s, S


# ----------------------------------------------------------------------------
# Stage 2: moments, nonparametric (exactly identified) estimator
# ----------------------------------------------------------------------------


def _moment_blocks(e, q, mask, v_w, a_w, weighted=True, q_w=None):
    """Per-policy b_i and A_i such that m_i(eta) = b_i - A_i eta.

    Ordering of eta: (gamma(0), a, gamma(1), ..., gamma(T-1)).
    Weights: w_itk = 1/(s_it^2 s_i,t+k^2), omega_it = 1/s_it^4, s^2 = a q_w + v
    (q_w defaults to q; it is held fixed when differentiating wrt beta).
    """
    I, T = e.shape
    K = 2 + (T - 1)
    m = mask.astype(float)
    if q_w is None:
        q_w = q
    if weighted:
        s2 = np.maximum(a_w * q_w + v_w, 1e-8)
        om = m / s2 ** 2
    else:
        s2 = np.ones_like(q)
        om = m.copy()
    b = np.zeros((I, K))
    A = np.zeros((I, K))       # diagonal entries for lag rows
    Ad = np.zeros((I, 2, 2))   # 2x2 block for (gamma0, a)
    e2 = e ** 2
    b[:, 0] = (om * e2).sum(1)
    b[:, 1] = (om * q * e2).sum(1)
    Ad[:, 0, 0] = om.sum(1)
    Ad[:, 0, 1] = Ad[:, 1, 0] = (om * q).sum(1)
    Ad[:, 1, 1] = (om * q * q).sum(1)
    for k in range(1, T):
        w = m[:, :-k] * m[:, k:] / (s2[:, :-k] * s2[:, k:])
        b[:, 1 + k] = (w * e[:, :-k] * e[:, k:]).sum(1)
        A[:, 1 + k] = w.sum(1)
    return b, A, Ad


def _abar(A, Ad):
    K = A.shape[1]
    M = np.zeros((K, K))
    M[:2, :2] = Ad.mean(0)
    for k in range(2, K):
        M[k, k] = A[:, k].mean()
    return M


def _contrib(b, A, Ad, eta):
    m = b.copy()
    m[:, :2] -= np.einsum('ijk,k->ij', Ad, eta[:2])
    m[:, 2:] -= A[:, 2:] * eta[2:]
    return m


def np_estimate(X, Z, mask, beta, p, weighted=True, n_iter=3, stage1=True):
    """Nonparametric moment estimator of eta with stage-1-corrected covariance.

    Returns eta_hat, its covariance V (already divided by I), and helpers.
    The weights are re-computed n_iter times from the current (gamma(0), a).
    """
    T = mask.shape[1]
    mu = np.exp(X @ beta)
    muT, mu_next = mu[:, :T], mu[:, T]
    ZT = Z[:, :T]
    e = np.where(mask, ZT / muT - 1, 0.0)
    q = muT ** (p - 2)
    v_w, a_w = 1.0, 1.0
    if weighted:
        # crude start: unweighted pass
        b, A, Ad = _moment_blocks(e, q, mask, 0, 0, weighted=False)
        eta = np.linalg.solve(_abar(A, Ad), b.mean(0))
        v_w, a_w = max(eta[0], 0.02), max(eta[1], 0.02)
    iters = n_iter if weighted else 1
    for _ in range(iters):
        b, A, Ad = _moment_blocks(e, q, mask, v_w, a_w, weighted)
        Abar = _abar(A, Ad)
        eta = np.linalg.solve(Abar, b.mean(0))
        if weighted:
            v_w, a_w = max(eta[0], 0.02), max(eta[1], 0.02)
    I = X.shape[0]
    m = _contrib(b, A, Ad, eta)
    if stage1:
        # derivative of the averaged moment wrt beta, weights held fixed
        Xtr = X[:, :T]
        d = len(beta)
        G = np.zeros((m.shape[1], d))
        h = 1e-5
        for j in range(d):
            vals = []
            for sgn in (1, -1):
                bb = beta.copy()
                bb[j] += sgn * h
                mu_h = np.exp(Xtr @ bb)
                e_h = np.where(mask, ZT / mu_h - 1, 0.0)
                q_h = mu_h ** (p - 2)
                b_h, A_h, Ad_h = _moment_blocks(e_h, q_h, mask, v_w, a_w, weighted, q_w=q)
                vals.append(_contrib(b_h, A_h, Ad_h, eta).mean(0))
            G[:, j] = (vals[0] - vals[1]) / (2 * h)
        s, S = glm_score_parts(Xtr, ZT, mask, beta, p)
        m = m - s @ np.linalg.solve(S, G.T)
    m = m - m.mean(0)
    Omega = m.T @ m / I
    Ainv = np.linalg.inv(Abar)
    V = Ainv @ Omega @ Ainv.T / I
    return dict(eta=eta, V=V, e=e, q=q, mu=muT, mu_next=mu_next, v_w=v_w, a_w=a_w)


# ----------------------------------------------------------------------------
# Parametric latent structures and minimum-distance (efficient GMM) fit
# ----------------------------------------------------------------------------


class Structure:
    """gamma(k; psi) and the nugget a, with an unconstrained parametrization."""

    def __init__(self, name, npar, gam, to_nat, start):
        self.name, self.npar, self.gam, self.to_nat, self.start = name, npar, gam, to_nat, start

    def eta(self, nat, T):
        g = self.gam(np.arange(T), nat)
        return np.concatenate([[g[0], nat[-1]], g[1:]])


S1 = Structure(  # theta-scale AR(1): gamma(k) = v rho^k
    'S1', 3,
    lambda k, n: n[0] * n[1] ** k,
    lambda u: np.array([np.exp(u[0]), expit(u[1]), np.exp(u[2])]),
    lambda eta: np.array([np.log(max(eta[0], 1e-3)),
                          logit(np.clip(eta[2] / max(eta[0], 1e-3), 0.05, 0.95)),
                          np.log(max(eta[1], 1e-3))]))

S2 = Structure(  # log-AR(1): gamma(k) = exp(s2w rhow^k) - 1
    'S2', 3,
    lambda k, n: np.expm1(n[0] * n[1] ** k),
    lambda u: np.array([np.exp(u[0]), expit(u[1]), np.exp(u[2])]),
    lambda eta: np.array([np.log(np.log1p(max(eta[0], 1e-3))),
                          logit(np.clip(np.log1p(max(eta[2], 1e-4)) / np.log1p(max(eta[0], 1e-3)), 0.05, 0.95)),
                          np.log(max(eta[1], 1e-3))]))

S3 = Structure(  # exchangeable (static): gamma(k) = v
    'S3', 2,
    lambda k, n: n[0] * np.ones_like(np.asarray(k, float)),
    lambda u: np.array([np.exp(u[0]), np.exp(u[1])]),
    lambda eta: np.array([np.log(max(np.mean(eta[2:]), 1e-3)), np.log(max(eta[1], 1e-3))]))


def md_fit(struct, eta_hat, V, idx=None):
    """Minimum-distance fit Q(psi) = (eta_hat - eta(psi))' V^{-1} (eta_hat - eta(psi)).

    Because the moment conditions are linear in eta, this is the efficient
    two-step GMM estimator, and Q at the minimum is Hansen's J statistic.
    idx selects a subset of eta (e.g. lags only).
    """
    T = len(eta_hat) - 1
    if idx is None:
        idx = np.arange(len(eta_hat))
    Vi = np.linalg.inv(V[np.ix_(idx, idx)])
    eh = eta_hat[idx]

    def Q(u):
        with np.errstate(over='ignore', invalid='ignore'):
            d = eh - struct.eta(struct.to_nat(u), T)[idx]
            val = d @ Vi @ d
        return val if np.isfinite(val) else 1e300

    u0 = struct.start(eta_hat)
    best = None
    for shift in (0.0, 1.0, -1.0):
        st = u0.copy()
        if struct.npar == 3:
            st[1] = st[1] + shift
        res = minimize(Q, st, method='BFGS', options=dict(gtol=1e-8, maxiter=500))
        if best is None or res.fun < best.fun:
            best = res
    nat = struct.to_nat(best.x)
    # sandwich (efficient) covariance in the natural parametrization
    H = _num_jac(lambda n: struct.eta(n, T)[idx], nat)
    try:
        Vnat = np.linalg.inv(H.T @ Vi @ H)
    except np.linalg.LinAlgError:
        Vnat = np.full((len(nat), len(nat)), np.nan)
    df = len(idx) - struct.npar
    return dict(nat=nat, Vnat=Vnat, J=best.fun, df=df,
                pJ=chi2.sf(best.fun, df) if df > 0 else np.nan)


def _num_jac(f, x, h=1e-6):
    f0 = f(x)
    J = np.zeros((len(f0), len(x)))
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        hj = h * max(1.0, abs(x[j]))
        xp[j] += hj
        xm[j] -= hj
        J[:, j] = (f(xp) - f(xm)) / (2 * hj)
    return J


def wald_tests(eta_hat, V):
    """Static (gamma(1)=...=gamma(T-1)) and no-dependence (gamma(k)=0, k>=1) Wald tests."""
    L = len(eta_hat) - 2              # number of lags T-1
    g = eta_hat[2:]
    Vg = V[2:, 2:]
    W0 = g @ np.linalg.solve(Vg, g)
    out = dict(W_nodep=W0, p_nodep=chi2.sf(W0, L))
    if L >= 2:
        R = np.zeros((L - 1, L))
        for j in range(L - 1):
            R[j, j], R[j, j + 1] = 1, -1
        Rg = R @ g
        Ws = Rg @ np.linalg.solve(R @ Vg @ R.T, Rg)
        out.update(W_static=Ws, p_static=chi2.sf(Ws, L - 1))
    return out


# ----------------------------------------------------------------------------
# Premiums
# ----------------------------------------------------------------------------


BIG = 1e10


def bl_weights(gam, a, q, mask):
    """Theorem 1 weights w = (Gamma_T + D_T)^{-1} c_T for every policy.

    gam: array gamma(0..T); q: (I,T) values mu_t^(p-2); unobserved years get
    an (effectively) infinite nugget so that their weight is zero.
    """
    I, T = q.shape
    lag = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    G = gam[lag]
    d = np.where(mask, a * q, BIG)
    M = G[None, :, :] + d[:, :, None] * np.eye(T)[None]
    c = gam[T - np.arange(T)]          # (gamma(T), ..., gamma(1))
    w = np.linalg.solve(M, np.broadcast_to(c, (I, T))[..., None])[..., 0]
    return np.where(mask, w, 0.0)


def exch_weights(v, a, q, mask):
    """Corollary 2 (exchangeable) weights via Sherman-Morrison."""
    r = np.where(mask, v / (a * q), 0.0)
    return r / (1 + r.sum(1, keepdims=True))


def kalman_weights(v, rho, a, q, mask):
    """Proposition 2: updating-type recursion for the theta-scale AR(1) structure.

    Returns the implied weights w_t (computed by propagating unit vectors),
    the gains K_t and the running weight sum.
    """
    I, T = q.shape
    xhat = np.zeros((I, T))          # coefficients of e_1..e_T in xhat_t
    P = np.full(I, v)
    K = np.zeros((I, T))
    for t in range(T):
        d = np.where(mask[:, t], a * q[:, t], np.inf)
        Kt = np.where(mask[:, t], P / (P + d), 0.0)
        K[:, t] = Kt
        new = (1 - Kt)[:, None] * xhat
        new[:, t] += Kt
        xhat = rho * new
        P = rho ** 2 * (1 - Kt) * P + (1 - rho ** 2) * v
    return xhat, K


def premium(mu_next, w, e):
    return mu_next * (1 + (w * e).sum(1))


def tweedie_deviance(y, m, p):
    m = np.maximum(m, 1e-10)
    return 2 * (np.power(y, 2 - p) / ((1 - p) * (2 - p)) - y * m ** (1 - p) / (1 - p)
                + m ** (2 - p) / (2 - p))


# ----------------------------------------------------------------------------
# Exact population MSE of linear predictors (unit test of the simulation)
# ----------------------------------------------------------------------------


def exact_mse(gam, a, q, q_next, mu_next, w, mask):
    """E[(Z_{T+1} - P)^2 | x] for P = mu_{T+1}(1 + w'e), true moments (gam, a)."""
    I, T = q.shape
    lag = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    G = gam[lag]
    d = np.where(mask, a * q, 0.0)
    c = gam[T - np.arange(T)]
    quad = np.einsum('it,ts,is->i', w, G, w) + (w ** 2 * d).sum(1)
    return mu_next ** 2 * (gam[0] + a * q_next - 2 * w @ c + quad)


def best_exchangeable(gam, a, q, q_next, mu_next, mask):
    """Oracle-parameter best exchangeable (static) predictor: optimize the static v."""
    def obj(lv):
        w = exch_weights(np.exp(lv), a, q, mask)
        return exact_mse(gam, a, q, q_next, mu_next, w, mask).mean()
    res = minimize_scalar(obj, bounds=(np.log(1e-4), np.log(50)), method='bounded')
    return np.exp(res.x), res.fun
