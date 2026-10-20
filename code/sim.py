"""Simulation study (Studies A and B) for the revised manuscript.

Usage:  python3 sim.py [main|sens|all] [reps]
Writes one compressed .npz per design cell into ../results/.
"""

import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

import tweedie_gmm as tg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
BETA = np.array([2.5, 0.5, -0.5])
METHODS = ['naive', 'static', 'S1', 'S2', 'NP', 'S1lag', 'S1unw',
           'oracle_param', 'oracle_exch', 'oracle_theta']


def cell(name, v=0.5, r1=0.7, I=2000, T=5, phi=10.0, p=1.5, p_work=1.5,
         miss=0.0, latent='logar1', profile_p=False, beta=tuple(BETA)):
    return dict(name=name, v=v, r1=r1, I=I, T=T, phi=phi, p=p, p_work=p_work,
                miss=miss, latent=latent, profile_p=profile_p, beta=beta)


def main_cells():
    cells = []
    for v in (0.1, 0.25, 0.5, 1.0, 2.0):
        for r1 in (0.3, 0.5, 0.7, 0.9):
            cells.append(cell('grid_v%g_r%g' % (v, r1), v=v, r1=r1))
    return cells


def sens_cells():
    return [
        cell('sens_I500', I=500),
        cell('sens_I10000', I=10000),
        cell('sens_T3', T=3),
        cell('sens_T10', T=10),
        cell('sens_phi5', phi=5.0),
        cell('sens_phi20', phi=20.0),
        cell('sens_p1.3', p=1.3, profile_p=True),
        cell('sens_p1.7', p=1.7, profile_p=True),
        cell('sens_miss20', miss=0.2),
        cell('sens_hetero', beta=(2.0, 1.5, -0.5)),
        cell('null_static', r1=1.0, latent='static'),
        cell('null_indep', r1=0.0, latent='indep'),
    ]


def latent_params(c):
    if c['latent'] == 'static':
        return np.log1p(c['v']), 1.0
    if c['latent'] == 'indep':
        return np.log1p(c['v']), 0.0
    return tg.logar1_params(c['v'], c['r1'])


def truth(c):
    s2w, rhow = latent_params(c)
    gam = tg.gamma_logar1(np.arange(c['T'] + 1), s2w, rhow)
    a = c['phi'] * np.exp(s2w * c['p'] * (c['p'] - 1) / 2)
    return s2w, rhow, gam, a


# ----------------------------------------------------------------------------
# p profiling: minimise the J statistic of the diagonal moments
# ----------------------------------------------------------------------------


def diag_J(e, q, mu, mask, v_w, a_w):
    m = mask.astype(float)
    s2 = np.maximum(a_w * q + v_w, 1e-8)
    om = m / s2 ** 2
    z = np.stack([np.ones_like(q), q, q * np.log(mu)], axis=2)       # instruments
    reg = np.stack([np.ones_like(q), q], axis=2)                     # (gamma0, a)
    A = np.einsum('it,itj,itk->ijk', om, z, reg)                     # (I,3,2)
    b = np.einsum('it,itj,it->ij', om, z, e ** 2)                    # (I,3)
    Ab, bb = A.mean(0), b.mean(0)
    W = np.linalg.inv(np.cov(b.T) + 1e-12 * np.eye(3))
    for _ in range(2):
        th = np.linalg.solve(Ab.T @ W @ Ab, Ab.T @ W @ bb)
        mi = b - A @ th
        Om = np.cov(mi.T, bias=True)
        W = np.linalg.inv(Om)
    mbar = mi.mean(0)
    return len(e) * mbar @ W @ mbar


def profile_p(d, grid=np.round(np.arange(1.1, 1.91, 0.05), 2)):
    T = d['T']
    Js = []
    for p in grid:
        beta = tg.fit_glm(d['X'][:, :T], d['Z'][:, :T], d['mask'], p)
        est = tg.np_estimate(d['X'], d['Z'], d['mask'], beta, p, stage1=False)
        Js.append(diag_J(est['e'], est['q'], est['mu'], d['mask'], est['v_w'], est['a_w']))
    return grid[int(np.argmin(Js))]


# ----------------------------------------------------------------------------
# One replication
# ----------------------------------------------------------------------------


def _r1_and_v(struct_name, nat):
    if struct_name == 'S1':
        return nat[0], nat[1]
    if struct_name == 'S2':
        return np.expm1(nat[0]), np.expm1(nat[0] * nat[1]) / np.expm1(nat[0])
    return nat[0], 1.0


def _ci_cover(f, struct_name, v0, r10):
    """Wald CIs (log scale for v, natural scale for r1) via the delta method."""
    g = lambda n: np.array([np.log(_r1_and_v(struct_name, n)[0]), _r1_and_v(struct_name, n)[1]])
    J = tg._num_jac(g, f['nat'])
    Vd = J @ f['Vnat'] @ J.T
    est = g(f['nat'])
    se = np.sqrt(np.maximum(np.diag(Vd), 0))
    cov_v = abs(est[0] - np.log(v0)) <= 1.96 * se[0]
    cov_r = abs(est[1] - r10) <= 1.96 * se[1]
    return cov_v, cov_r, se


def one_rep(args):
    c, seed = args
    rng = np.random.default_rng(seed)
    T, p = c['T'], c['p']
    s2w, rhow, gam0, a0 = truth(c)
    r10 = gam0[1] / gam0[0]
    d = tg.simulate_panel(rng, I=c['I'], T=T, beta=c['beta'], phi=c['phi'], p=p,
                          s2w=s2w, rhow=rhow, miss=c['miss'])
    X, Z, mask = d['X'], d['Z'], d['mask']
    out = dict(zero_share=np.mean(Z[:, :T][mask] == 0))

    pw = c['p_work']
    if c['profile_p']:
        pw_hat = profile_p(d)
        out['p_hat'] = pw_hat
    p_list = [('', pw)] + ([('_pprof', pw_hat)] if c['profile_p'] else [])

    mu_true = d['mu']
    theta_next = d['theta'][:, T]
    Znext = Z[:, T]
    target = mu_true[:, T] * theta_next
    e_true = np.where(mask, Z[:, :T] / mu_true[:, :T] - 1, 0.0)
    q_true = mu_true[:, :T] ** (p - 2)

    def record(tag, P):
        nonpos = np.sum(P <= 0)
        out['pe_' + tag] = np.mean((P - target) ** 2)
        out['mse_' + tag] = np.mean((Znext - P) ** 2)
        out['dev_' + tag] = np.mean(tg.tweedie_deviance(Znext, np.maximum(P, 1e-6 * mu_true[:, T]), p))
        out['nonpos_' + tag] = nonpos

    # oracle predictors (true beta, true moments)
    w_op = tg.bl_weights(gam0, a0, q_true, mask)
    record('oracle_param', tg.premium(mu_true[:, T], w_op, e_true))
    record('oracle_theta', target)
    if c['latent'] != 'static':
        v_ex, _ = tg.best_exchangeable(gam0, a0, q_true, mu_true[:, T] ** (p - 2), mu_true[:, T], mask)
    else:
        v_ex = gam0[0]
    record('oracle_exch', tg.premium(mu_true[:, T], tg.exch_weights(v_ex, a0, q_true, mask), e_true))

    for suffix, pw_ in p_list:
        beta = tg.fit_glm(X[:, :T], Z[:, :T], mask, pw_)
        est = tg.np_estimate(X, Z, mask, beta, pw_)
        eta, V = est['eta'], est['V']
        e, q, mu_next = est['e'], est['q'], est['mu_next']
        record('naive' + suffix, mu_next)

        # nonparametric gamma with a geometric tail for gamma(T)
        g = np.concatenate([[eta[0]], eta[2:]])
        tail = 0.0
        if T >= 3 and g[-2] > 0 and g[-1] > 0:
            tail = g[-1] * min(g[-1] / g[-2], 1.0)
        elif T == 2:
            tail = max(g[-1], 0.0) ** 2 / max(g[0], 1e-6)
        g_np = np.concatenate([g, [tail]])
        record('NP' + suffix, tg.premium(mu_next, tg.bl_weights(g_np, eta[1], q, mask), e))
        if not suffix:
            out['np_v'], out['np_a'] = eta[0], eta[1]
            out['np_r1'] = eta[2] / eta[0] if eta[0] > 0 else np.nan
            out['np_g1'] = eta[2]
            # NP coverage for gamma(1) (natural scale)
            out['np_cov_g1'] = abs(eta[2] - gam0[1]) <= 1.96 * np.sqrt(V[2, 2])
            out.update(tg.wald_tests(eta, V))

        for S in (tg.S1, tg.S2, tg.S3):
            f = tg.md_fit(S, eta, V)
            gam = S.gam(np.arange(T + 1), f['nat'])
            a = f['nat'][-1]
            if S.name == 'S1':
                w, _ = tg.kalman_weights(f['nat'][0], f['nat'][1], a, q, mask)
            elif S.name == 'S3':
                w = tg.exch_weights(f['nat'][0], a, q, mask)
            else:
                w = tg.bl_weights(gam, a, q, mask)
            tag = {'S3': 'static'}.get(S.name, S.name) + suffix
            record(tag, tg.premium(mu_next, w, e))
            if not suffix:
                vh, r1h = _r1_and_v(S.name, f['nat'])
                out[S.name + '_v'], out[S.name + '_r1'], out[S.name + '_a'] = vh, r1h, a
                out[S.name + '_pJ'] = f['pJ']
                if S.name != 'S3':
                    cv, cr, se = _ci_cover(f, S.name, gam0[0], r10)
                    out[S.name + '_cov_v'], out[S.name + '_cov_r1'] = cv, cr
                    out[S.name + '_se_r1'] = se[1]
                    out[S.name + '_boundary'] = (f['nat'][1] > 0.99) or (vh < 1e-3)

        if not suffix:
            # lag-only variant: (v, rho) from the lag moments, a from the NP diagonal block
            f = tg.md_fit(tg.S1, eta, V, idx=np.arange(1, len(eta)))
            w, _ = tg.kalman_weights(f['nat'][0], f['nat'][1], f['nat'][2], q, mask)
            record('S1lag', tg.premium(mu_next, w, e))
            out['S1lag_v'], out['S1lag_r1'] = f['nat'][0], f['nat'][1]
            # equal-weight (unweighted) moments
            est_u = tg.np_estimate(X, Z, mask, beta, pw_, weighted=False)
            f = tg.md_fit(tg.S1, est_u['eta'], est_u['V'])
            w, _ = tg.kalman_weights(f['nat'][0], f['nat'][1], f['nat'][2], q, mask)
            record('S1unw', tg.premium(mu_next, w, e))
            out['S1unw_v'], out['S1unw_r1'] = f['nat'][0], f['nat'][1]
    return out


# ----------------------------------------------------------------------------
# Exact population quantities for a cell (integrating over covariates)
# ----------------------------------------------------------------------------


def exact_cell(c, n=2000000, seed=12345):
    rng = np.random.default_rng(seed)
    T, p = c['T'], c['p']
    s2w, rhow, gam0, a0 = truth(c)
    d = tg.simulate_panel(rng, I=n, T=T, beta=c['beta'], phi=c['phi'], p=p, s2w=s2w, rhow=rhow, miss=c['miss'])
    mu, mask = d['mu'], d['mask']
    q, qn, mn = mu[:, :T] ** (p - 2), mu[:, T] ** (p - 2), mu[:, T]
    noise = (mn ** 2 * a0 * qn).mean()
    zero = np.zeros((n, T))
    res = dict(mse_naive=tg.exact_mse(gam0, a0, q, qn, mn, zero, mask).mean())
    w = tg.bl_weights(gam0, a0, q, mask)
    res['mse_oracle_param'] = tg.exact_mse(gam0, a0, q, qn, mn, w, mask).mean()
    if c['latent'] != 'static':
        _, res['mse_oracle_exch'] = tg.best_exchangeable(gam0, a0, q, qn, mn, mask)
    else:
        res['mse_oracle_exch'] = res['mse_oracle_param']
    res['noise'] = noise
    for k in list(res):
        if k.startswith('mse_'):
            res['pe_' + k[4:]] = res[k] - noise
    res['gamma'] = gam0
    res['a'] = a0
    res['s2w'], res['rhow'] = s2w, rhow
    return res


def run(cells, reps, procs=4):
    os.makedirs(OUT, exist_ok=True)
    with Pool(procs) as pool:
        for c in cells:
            t0 = time.time()
            ss = np.random.SeedSequence(zlib.crc32(c['name'].encode()))
            seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(reps)]
            res = pool.map(one_rep, [(c, s) for s in seeds], chunksize=4)
            keys = sorted(set().union(*[r.keys() for r in res]))
            arr = {k: np.array([r.get(k, np.nan) for r in res], dtype=float) for k in keys}
            ex = exact_cell(c)
            np.savez_compressed(os.path.join(OUT, c['name'] + '.npz'),
                                cell=np.array([repr(c)]),
                                **arr, **{'exact_' + k: np.asarray(v) for k, v in ex.items()})
            print('%-18s %4d reps  %.0fs' % (c['name'], reps, time.time() - t0), flush=True)


# ----------------------------------------------------------------------------
# Cluster (policyholder) bootstrap: percentile intervals and bias correction
# ----------------------------------------------------------------------------


def boot_rep(args):
    """Same data as one_rep(c, seed); B policy-level resamples refit stages 1-2.

    The minimum-distance weight matrix is held at its full-sample value
    (fixed-weight bootstrap), so each resample costs one GLM fit and one
    closed-form moment pass.
    """
    c, seed, B = args
    rng = np.random.default_rng(seed)
    T, p = c['T'], c['p']
    s2w, rhow, gam0, a0 = truth(c)
    r10 = gam0[1] / gam0[0]
    d = tg.simulate_panel(rng, I=c['I'], T=T, beta=np.asarray(c['beta']), phi=c['phi'], p=p,
                          s2w=s2w, rhow=rhow, miss=c['miss'])
    X, Z, mask = d['X'], d['Z'], d['mask']
    beta = tg.fit_glm(X[:, :T], Z[:, :T], mask, c['p_work'])
    est = tg.np_estimate(X, Z, mask, beta, c['p_work'])
    f = tg.md_fit(tg.S2, est['eta'], est['V'])
    v_hat, r_hat = _r1_and_v('S2', f['nat'])
    brng = np.random.default_rng(seed + 1)
    vs, rs = [], []
    for _ in range(B):
        idx = brng.integers(0, c['I'], c['I'])
        Xb, Zb, mb = X[idx], Z[idx], mask[idx]
        bb = tg.fit_glm(Xb[:, :T], Zb[:, :T], mb, c['p_work'])
        eb = tg.np_estimate(Xb, Zb, mb, bb, c['p_work'], stage1=False)
        fb = tg.md_fit(tg.S2, eb['eta'], est['V'])
        vb, rb = _r1_and_v('S2', fb['nat'])
        vs.append(vb)
        rs.append(rb)
    vs, rs = np.array(vs), np.array(rs)
    lo_v, hi_v = np.percentile(vs, [2.5, 97.5])
    lo_r, hi_r = np.percentile(rs, [2.5, 97.5])
    return dict(v_hat=v_hat, r_hat=r_hat, v_bc=2 * v_hat - vs.mean(), r_bc=2 * r_hat - rs.mean(),
                cov_v=float(lo_v <= gam0[0] <= hi_v), cov_r=float(lo_r <= r10 <= hi_r),
                cov_v_wald=float(_ci_cover(f, 'S2', gam0[0], r10)[0]),
                cov_r_wald=float(_ci_cover(f, 'S2', gam0[0], r10)[1]))


def run_boot(names, reps=200, B=199, procs=4):
    cells = {c['name']: c for c in main_cells()}
    with Pool(procs) as pool:
        for nm in names:
            c = cells[nm]
            t0 = time.time()
            ss = np.random.SeedSequence(zlib.crc32(c['name'].encode()))
            seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(reps)]
            res = pool.map(boot_rep, [(c, s, B) for s in seeds], chunksize=2)
            arr = {k: np.array([r[k] for r in res]) for k in res[0]}
            np.savez_compressed(os.path.join(OUT, 'boot_' + nm + '.npz'), **arr)
            print('boot %-14s %d reps  %.0fs' % (nm, reps, time.time() - t0), flush=True)


def update_exact(cells):
    """Recompute the exact population quantities of saved cells (no re-simulation)."""
    for c in cells:
        pth = os.path.join(OUT, c['name'] + '.npz')
        arr = dict(np.load(pth))
        ex = exact_cell(c)
        arr.update({'exact_' + k: np.asarray(v) for k, v in ex.items()})
        np.savez_compressed(pth, **arr)
        print('exact %-18s done' % c['name'], flush=True)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which == 'exact':
        update_exact(main_cells() + sens_cells())
        sys.exit()
    if which == 'boot':
        reps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        run_boot(sys.argv[3].split(','), reps=reps)
        sys.exit()
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    cells = {'main': main_cells(), 'sens': sens_cells(),
             'all': main_cells() + sens_cells()}[which]
    if len(sys.argv) > 3:
        cells = [c for c in cells if c['name'] in sys.argv[3].split(',')]
    run(cells, reps)
