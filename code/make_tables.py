"""Summarize ../results/*.npz into LaTeX tables (../results/tables) and figures (../figures)."""

import os
import json

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
TAB = os.path.join(RES, 'tables')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

V_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)
R_GRID = (0.3, 0.5, 0.7, 0.9)
rng = np.random.default_rng(0)


def load(name):
    return dict(np.load(os.path.join(RES, name + '.npz')))


def gname(v, r):
    return 'grid_v%g_r%g' % (v, r)


def rel_red(r, m, base='naive', key='mse'):
    """Per-replication relative reduction (%) of key vs base: mean and MC SE."""
    x = 100 * (r['%s_%s' % (key, base)] - r['%s_%s' % (key, m)]) / r['%s_%s' % (key, base)]
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def share(r, m, base='naive', oracle='oracle_param', B=1000):
    """Share of oracle gain on the premium-error scale (ratio of means) with bootstrap SE."""
    a, b, c = r['pe_' + base], r['pe_' + m], r['pe_' + oracle]
    est = (a.mean() - b.mean()) / (a.mean() - c.mean())
    n = len(a)
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bs.append((a[i].mean() - b[i].mean()) / (a[i].mean() - c[i].mean()))
    return 100 * est, 100 * np.std(bs, ddof=1)


def f(x, d=2):
    s = ('%.' + str(d) + 'f') % x
    return '$-$' + s[1:] if s.startswith('-') else s


def fs(x):
    """Format a share (%); very negative shares (tiny oracle gain) are truncated."""
    return '$<-100$' if x < -100 else f(x, 0)


def pm(m, s, d=2):
    return '%s (%s)' % (f(m, d), f(s, d))


def rej(x):
    x = x[np.isfinite(x)]
    return f(100 * np.mean(x < 0.05), 1)


def med_iqr(x, d=2):
    x = x[np.isfinite(x)]
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return '%s [%s, %s]' % (f(q2, d), f(q1, d), f(q3, d))


def write(name, body):
    with open(os.path.join(TAB, name + '.tex'), 'w') as fh:
        fh.write(body)


summary = {}

# ----------------------------------------------------------------------------
# Table: exact population MSE (unit test) and oracle gains
# ----------------------------------------------------------------------------
rows = []
for v in V_GRID:
    for r1 in R_GRID:
        r = load(gname(v, r1))
        ex_n, ex_op, ex_be = r['exact_mse_naive'], r['exact_mse_oracle_param'], r['exact_mse_oracle_exch']
        sim = r['mse_oracle_param']
        z = (sim.mean() - ex_op) / (sim.std(ddof=1) / np.sqrt(len(sim)))
        rows.append('%g & %g & %.0f & %s & %s & %s & %s & %s \\\\' % (
            v, r1, 100 * r['zero_share'].mean(), f(ex_n, 1), f(100 * (ex_n - ex_op) / ex_n),
            f(100 * (ex_n - ex_be) / ex_n), f(ex_op, 1), pm(sim.mean(), sim.std(ddof=1) / np.sqrt(len(sim)), 1)))
        summary[gname(v, r1)] = dict(z_unit=float(z), exact_gain_op=float(100 * (ex_n - ex_op) / ex_n),
                                     exact_gain_be=float(100 * (ex_n - ex_be) / ex_n),
                                     zero=float(r['zero_share'].mean()))
write('tab_exact', '\n'.join(rows) + '\n')

# ----------------------------------------------------------------------------
# Table: Study A point estimation (S1, S2)
# ----------------------------------------------------------------------------
rows = []
for v in V_GRID:
    for r1 in R_GRID:
        r = load(gname(v, r1))
        rows.append('%g & %g & %s & %s & %s & %s \\\\' % (
            v, r1, med_iqr(r['S1_v']), med_iqr(r['S1_r1']), med_iqr(r['S2_v']), med_iqr(r['S2_r1'])))
write('tab_estA', '\n'.join(rows) + '\n')

# Study A inference
rows = []
for v in V_GRID:
    for r1 in R_GRID:
        r = load(gname(v, r1))
        rows.append('%g & %g & %s & %s & %s & %s & %s & %s & %s & %s \\\\' % (
            v, r1, f(100 * np.mean(r['S2_cov_v']), 1), f(100 * np.mean(r['S2_cov_r1']), 1),
            f(100 * np.mean(r['np_cov_g1']), 1), rej(r['S1_pJ']), rej(r['S2_pJ']), rej(r['S3_pJ']),
            rej(r['p_static']), f(100 * np.mean(r['S1_boundary']), 1)))
        summary[gname(v, r1)].update(cov_v=float(np.mean(r['S2_cov_v'])), cov_r=float(np.mean(r['S2_cov_r1'])),
                                     J1=float(np.mean(r['S1_pJ'] < .05)), J2=float(np.mean(r['S2_pJ'] < .05)),
                                     static=float(np.mean(r['p_static'] < .05)),
                                     nodep=float(np.mean(r['p_nodep'] < .05)),
                                     boundary=float(np.mean(r['S1_boundary'])))
write('tab_infA', '\n'.join(rows) + '\n')

# ----------------------------------------------------------------------------
# Table: Study B prediction
# ----------------------------------------------------------------------------
rows = []
for v in V_GRID:
    for r1 in R_GRID:
        r = load(gname(v, r1))
        cells = [pm(*rel_red(r, m)) for m in ('static', 'S1', 'S2', 'oracle_param')]
        sh1, sh2 = share(r, 'S1'), share(r, 'S2')
        shs = share(r, 'static')
        beat = 100 * np.mean(r['pe_S1'] < r['pe_naive'])
        rows.append('%g & %g & %s & %s & %s & %s & %s \\\\' % (
            v, r1, ' & '.join(cells), fs(shs[0]), fs(sh1[0]), fs(sh2[0]), f(beat, 0)))
        summary[gname(v, r1)].update(
            red={m: rel_red(r, m) for m in ('static', 'S1', 'S2', 'NP', 'oracle_param', 'oracle_exch')},
            red_pe={m: rel_red(r, m, key='pe') for m in ('static', 'S1', 'S2', 'NP', 'oracle_param', 'oracle_exch')},
            share={m: share(r, m) for m in ('static', 'S1', 'S2', 'NP', 'S1lag', 'S1unw', 'oracle_exch')},
            beat=float(beat),
            nonpos={m: float(r['nonpos_' + m].sum()) for m in ('S1', 'S2', 'NP', 'static', 'S1lag', 'S1unw')},
            dev={m: float(r['dev_' + m].mean()) for m in ('naive', 'static', 'S1', 'S2', 'NP', 'oracle_param')})
write('tab_predB', '\n'.join(rows) + '\n')

# ----------------------------------------------------------------------------
# Table: estimator variants (share of oracle gain) and deviance
# ----------------------------------------------------------------------------
rows = []
for nm, lab in [(gname(v, 0.7), '%g & 0.7' % v) for v in V_GRID] + \
               [(gname(v, 0.9), '%g & 0.9' % v) for v in V_GRID] + [('sens_hetero', '\\multicolumn{2}{c}{hetero.}')]:
    r = load(nm)
    sh = [fs(share(r, m)[0]) for m in ('static', 'S1', 'S1lag', 'S1unw', 'S2', 'NP')]
    dv = [f(100 * (r['dev_naive'].mean() - r['dev_' + m].mean()) / r['dev_naive'].mean(), 2)
          for m in ('static', 'S1', 'NP')]
    rows.append('%s & %s & %s \\\\' % (lab, ' & '.join(sh), ' & '.join(dv)))
write('tab_variants', '\n'.join(rows) + '\n')

# ----------------------------------------------------------------------------
# Table: sensitivity and null designs
# ----------------------------------------------------------------------------
sens = [('grid_v0.5_r0.7', 'Central ($I=2000$, $T=5$, $\\phi=10$, $p=1.5$)', ''),
        ('sens_I500', '$I=500$', ''), ('sens_I10000', '$I=10{,}000$', ''),
        ('sens_T3', '$T=3$', ''), ('sens_T10', '$T=10$', ''),
        ('sens_phi5', '$\\phi=5$', ''), ('sens_phi20', '$\\phi=20$', ''),
        ('sens_p1.3', 'True $p=1.3$, working $p=1.5$', ''), ('sens_p1.3', 'True $p=1.3$, $p$ profiled', '_pprof'),
        ('sens_p1.7', 'True $p=1.7$, working $p=1.5$', ''), ('sens_p1.7', 'True $p=1.7$, $p$ profiled', '_pprof'),
        ('sens_miss20', '20\\% of years missing', ''), ('sens_hetero', 'Heterogeneous $\\mu$ ($\\beta_1=1.5$)', '')]
rows = []
for nm, lab, sfx in sens:
    r = load(nm)
    if sfx:
        est = '\\multicolumn{3}{c}{---}'
        base = 'naive_pprof'
        sh1 = share(r, 'S1_pprof', base=base)[0]
        sh2 = share(r, 'S2_pprof', base=base)[0]
        shs = share(r, 'static_pprof', base=base)[0]
        rr = rel_red(r, 'S1_pprof', base=base)
        ph = r['p_hat']
        lab = lab + ' (median $\\hat p$=%s)' % f(np.median(ph), 2)
        summary[nm + '_phat'] = dict(median=float(np.median(ph)), q1=float(np.percentile(ph, 25)),
                                     q3=float(np.percentile(ph, 75)))
    else:
        est = '%s & %s & %s' % (f(np.median(r['S1_v'])), f(np.median(r['S1_r1'])), rej(r['S2_pJ']))
        sh1, sh2, shs = share(r, 'S1')[0], share(r, 'S2')[0], share(r, 'static')[0]
        rr = rel_red(r, 'S1')
    rows.append('%s & %s & %s & %s & %s & %s \\\\' % (lab, est, fs(shs), fs(sh1), fs(sh2), pm(*rr)))
    summary.setdefault('sens', {})[nm + sfx] = dict(share_S1=float(sh1), share_S2=float(sh2), share_static=float(shs),
                                                   red_S1=rr)
write('tab_sens', '\n'.join(rows) + '\n')

# truth values for sensitivity cells (all share v=0.5, r1=0.7 except nulls)
rows = []
for nm, lab in [('null_static', 'Static truth ($\\gamma(k)=v$)'), ('null_indep', 'Independence ($\\gamma(k)=0$, $k\\ge1$)'),
                ('grid_v0.5_r0.7', 'Central design')]:
    r = load(nm)
    rows.append('%s & %s & %s & %s & %s & %s & %s & %s \\\\' % (
        lab, rej(r['p_static']), rej(r['p_nodep']), rej(r['S1_pJ']), rej(r['S2_pJ']), rej(r['S3_pJ']),
        f(share(r, 'static')[0], 0) if nm != 'null_indep' else '---',
        f(share(r, 'S1')[0], 0) if nm != 'null_indep' else '---'))
    summary[nm + '_tests'] = dict(static=float(np.mean(r['p_static'] < .05)), nodep=float(np.mean(r['p_nodep'] < .05)),
                                  J1=float(np.mean(r['S1_pJ'] < .05)), J2=float(np.mean(r['S2_pJ'] < .05)),
                                  J3=float(np.mean(r['S3_pJ'] < .05)))
    if nm == 'null_indep':
        summary['null_indep_pred'] = {m: rel_red(r, m) for m in ('static', 'S1', 'S2', 'NP')}
write('tab_null', '\n'.join(rows) + '\n')

# bootstrap study
rows = []
for v, r1 in [(0.1, 0.3), (0.5, 0.7), (2.0, 0.9)]:
    pth = os.path.join(RES, 'boot_%s.npz' % gname(v, r1))
    if not os.path.exists(pth):
        continue
    b = dict(np.load(pth))
    rows.append('%g & %g & %s & %s & %s & %s & %s & %s \\\\' % (
        v, r1, f(100 * b['cov_v_wald'].mean(), 1), f(100 * b['cov_v'].mean(), 1),
        f(100 * b['cov_r_wald'].mean(), 1), f(100 * b['cov_r'].mean(), 1),
        f(np.median(b['v_hat'])), f(np.median(b['v_bc']))))
    summary['boot_' + gname(v, r1)] = {k: float(np.mean(b[k])) for k in ('cov_v', 'cov_r', 'cov_v_wald', 'cov_r_wald')}
write('tab_boot', '\n'.join(rows) + '\n')

with open(os.path.join(RES, 'summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o))

# ----------------------------------------------------------------------------
# Figures (static, print): palette from the reference instance
# ----------------------------------------------------------------------------
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
       '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})

# Figure 1: heatmap of the share of oracle gain captured by S1
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('seqblue', SEQ)
M = np.array([[summary[gname(v, r1)]['share']['S1'][0] for r1 in R_GRID] for v in V_GRID])
fig, ax = plt.subplots(figsize=(4.6, 3.4))
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect='auto', origin='lower')
for i in range(len(V_GRID)):
    for j in range(len(R_GRID)):
        ax.text(j, i, '%.0f%%' % M[i, j], ha='center', va='center',
                color='white' if M[i, j] > 62 else INK, fontsize=9)
ax.set_xticks(range(len(R_GRID)), [str(r) for r in R_GRID])
ax.set_yticks(range(len(V_GRID)), [str(v) for v in V_GRID])
ax.set_xlabel('lag-1 latent correlation $r(1)$')
ax.set_ylabel(r'latent variance $v=\mathrm{Var}(\theta)$')
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label('share of oracle gain captured (%)', color=INK)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_share_heatmap.pdf'))
plt.close(fig)

# Figure 2: relative premium-error reduction vs r(1)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for k, v in enumerate(V_GRID):
    y1 = [summary[gname(v, r)]['red_pe']['S1'][0] for r in R_GRID]
    y0 = [summary[gname(v, r)]['red_pe']['oracle_param'][0] for r in R_GRID]
    ax.plot(R_GRID, y0, color=SERIES[k], lw=1.2, ls='--')
    ax.plot(R_GRID, y1, color=SERIES[k], lw=2, marker='o', ms=4, label='$v=%g$' % v)
    ax.text(R_GRID[-1] + 0.015, y1[-1], '$v=%g$' % v, color=INK2, va='center', fontsize=8)
ax.set_xlim(0.27, 0.99)
ax.set_xticks(R_GRID)
ax.set_xlabel('lag-1 latent correlation $r(1)$')
ax.set_ylabel('reduction in premium error vs naive (%)')
ax.grid(axis='y', color=GRID, lw=0.6)
ax.set_axisbelow(True)
h, l = ax.get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], color=INK2, lw=2, marker='o', ms=4), Line2D([], [], color=INK2, lw=1.2, ls='--')]
l += ['S1 (estimated)', 'oracle parameters']
ax.legend(h, l, loc='upper left', fontsize=7.5, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_gain_vs_r1.pdf'))
plt.close(fig)

# Figure 3: sampling distributions of v-hat/v and r1-hat (r1 = 0.7 column)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
pos = np.arange(len(V_GRID))
for j, (key, lab, truthf) in enumerate([('v', r'$\hat v/v$', lambda v: v), ('r1', r'$\hat r(1)$', None)]):
    ax = axes[j]
    for k, (S, col) in enumerate([('S1', SERIES[0]), ('S2', SERIES[1])]):
        data = []
        for v in V_GRID:
            x = load(gname(v, 0.7))[S + '_' + key]
            data.append(x / v if key == 'v' else x)
        bp = ax.boxplot(data, positions=pos + (k - 0.5) * 0.32, widths=0.26, patch_artist=True,
                        showfliers=False, whis=(5, 95), medianprops=dict(color=INK, lw=1.2))
        for b in bp['boxes']:
            b.set(facecolor=col, edgecolor=col, alpha=0.85)
        for w in bp['whiskers'] + bp['caps']:
            w.set(color=col, lw=1)
    ax.axhline(1.0 if key == 'v' else 0.7, color=INK2, lw=1, ls=':')
    ax.set_xticks(pos, [str(v) for v in V_GRID])
    ax.set_xlabel('latent variance $v$')
    ax.set_ylabel(lab)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
from matplotlib.patches import Patch
axes[0].legend([Patch(color=SERIES[0]), Patch(color=SERIES[1])], [r'S1 ($\theta$-AR(1))', 'S2 (log-AR(1))'],
               loc='upper right', fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_estimates_box.pdf'))
plt.close(fig)
print('done')

# ----------------------------------------------------------------------------
# Inline the tables into the manuscript between %BEGIN/%END markers
# ----------------------------------------------------------------------------
import re
TEX = os.path.join(HERE, '..', 'main_ver1.tex')
src = open(TEX).read()
for fn in sorted(os.listdir(TAB)):
    nm = fn[:-4]
    body = open(os.path.join(TAB, fn)).read()
    src = re.sub(r'%%BEGIN %s\n.*?%%END %s\n' % (nm, nm),
                 lambda m: '%%BEGIN %s\n%s%%END %s\n' % (nm, body, nm), src, flags=re.S)
open(TEX, 'w').write(src)
print('tables inlined')
