"""Figures (coefficient / dot-and-whisker plots, 3 panels: index f | W | dW), each with a footnote that
defines the scenario, the arms and the metrics.
* --stat r2 (default, -> figures/): R2 of the index (corr^2 with true f), out-of-sample R2 against OBSERVED W and
  dW (within city-date / city); dashed line = ceiling (true E[W|x] as predictor), per sample size.
* --stat rmse (-> figures/rmse/): RMSE vs the true conditional mean, log axis, grey bar = pseudo-true.
* fig_<exp>_<scenario>: rows = arms; colour = sample size (one-hue ramp, light = small, dark = large);
  marker = head (filled = oracle / index, open = CV B-spline, diamond = native FE/MM); whiskers = +-1.96 MCSE.
* fig_overview_<exp>_<C>x<n>: rows = scenarios of the experiment, colour + marker = key arm
  (RNC_S, RNC, SUR_S, FE, MM); CV-spline head (R2) or oracle head (RMSE).
* fig_omega_*: E2 bias-variance curve over omega.
python -m minimal_e1.paper.figures --results DIR --out DIR
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import experiments as ex
from . import pseudotrue as pt
from . import summarize as sm

R2_FLOOR = -0.1                     # R2 axes run from R2_FLOOR to R2_TOP; lower means drawn as a '<' at the edge
R2_TOP = 1.05
R2_BELOW_NOTE = '◄ at the left edge = mean R² below %.1f (value not shown).' % R2_FLOOR

HERE = Path(__file__).resolve().parent
COLORS = {'RNC': '#0072B2', 'RNC_S': '#D55E00', 'RNC_P0.9': '#E69F00', 'RNC_P0.5': '#F0C27B', 'RNC_P0.99': '#CC8400',
          'RNC_SEL': '#8C564B', 'RNC_S_DW': '#9467BD', 'SUR': '#56B4E9', 'SUR_S': '#CC79A7', 'MRC_S': '#009E73',
          'FE': '#444444', 'FE_S': '#999999', 'NS': '#B0B0B0', 'MM': '#117733', 'ML': '#000000'}
PANELS = {
    'rmse': [('f', 'RMSE of standardised index f'), ('W', 'RMSE of W vs E[W|x]'), ('dW', 'RMSE of dW vs dE[W|x]')],
    'r2': [('f', 'R\u00b2 of index f (corr\u00b2 with true f)'), ('W', 'R\u00b2 of observed W (within city-date)'),
           ('dW', 'R\u00b2 of observed dW (within city)')],
}

def pt_lookup(exp, scenario, pt_dir, C=None, n=None):
    spec = ex.SCENARIOS[exp][scenario]
    for path in sorted(Path(pt_dir).glob(f'pt_{pt.pt_key(spec)}_C*_n*.json')):
        return json.loads(path.read_text())
    return None


# ---------------------------------------------------------------------------
# coefficient (dot-and-whisker) plots
# ---------------------------------------------------------------------------
# sample size = ordered magnitude -> one-hue ordinal ramp (validated, light -> dark = small -> large N)
SIZE_RAMP = {2: ['#6da7ec', '#0d366b'], 3: ['#6da7ec', '#256abf', '#0d366b'],
             5: ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']}
# head = secondary encoding by marker shape (filled = oracle/index, open = CV spline, diamond = native)
HEAD_MARK = {'index': dict(marker='o', fill=True), 'oracle': dict(marker='o', fill=True),
             'spline': dict(marker='o', fill=False), 'iso': dict(marker='s', fill=False),
             'native': dict(marker='D', fill=False)}
HEAD_LABEL = {'index': 'index (no head)', 'oracle': 'oracle head', 'spline': 'CV B-spline head',
              'iso': 'isotonic head', 'native': 'native (FE/MM)'}
HEADS_SHOWN = ['oracle', 'spline', 'native']
# arms top -> bottom: stability arms, unrestricted ordinal, practitioner, ML
ARM_ROWS = ['RNC_S', 'RNC_P0.9', 'RNC_P0.5', 'RNC_P0.99', 'RNC_SEL', 'RNC_S_DW', 'SUR_S', 'MRC_S', 'FE_S',
            'RNC', 'SUR', 'FE', 'NS', 'MM', 'ML']
GROUP_BREAKS = {'RNC', 'FE'}                     # thin rule above these rows
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
PT_KEY = {'f': 'rmse_f', 'W': 'rmse_W_oracle', 'dW': 'rmse_dW_oracle'}

# ---------------------------------------------------------------------------
# footnotes (definitions; full version in paper/DESIGN.md)
# ---------------------------------------------------------------------------
ARM_NOTES = {
    'RNC': 'RNC = RankNet, logistic pairwise loss within city-date cells, per-cell temperatures; no stability',
    'RNC_S': 'RNC_S = RNC with the HARD stability restriction (index confined to directions with zero link gaps)',
    'RNC_P0.9': 'RNC_P0.9 = RNC + stability PENALTY, links supply 90% of the curvature',
    'RNC_P0.5': 'RNC_P0.5 = penalty, 50% curvature share',
    'RNC_P0.99': 'RNC_P0.99 = penalty, 99% curvature share',
    'RNC_SEL': 'RNC_SEL = penalty share chosen by held-out Kendall tau',
    'RNC_S_DW': 'RNC_S_DW = RNC_S with distance-weighted pairs',
    'SUR': 'SUR = smoothed maximum rank correlation (Abrevaya), bandwidth n^-0.6; no stability',
    'SUR_S': 'SUR_S = SUR with the hard restriction',
    'MRC_S': 'MRC_S = exact maximum rank correlation with the hard restriction',
    'FE': 'FE = within-city-date OLS with city-date fixed effects (practitioner default)',
    'FE_S': 'FE_S = FE with the hard restriction',
    'NS': 'NS = FE on within-cell normal scores of W',
    'MM': 'MM = mixed model (random cell intercept and slope)',
    'ML': 'ML = exact maximum likelihood of the true model (only where correctly specified)',
}
SCENARIO_NOTES = {
    'E1_a': 'E1_a: additive map W = a_ct + f + noise (FE correctly specified); exact temporal + cross-city links',
    'E1_b': 'E1_b: affine map W = a_ct + b_ct (f + noise) (city-date slopes; mixed-effects world); exact links',
    'E1_c': 'E1_c: nonlinear city-varying convex/concave sinh map; FE misspecified; exact links',
    'E1_cD': 'E1_cD: map c with noise inside AND skewed noise outside the map; exact links',
    'L_none': 'E2 L_none: map c, no links declared (restricted arms not applicable)',
    'L_temp': 'E2 L_temp: map c, temporal links only (same stable tract at both dates)',
    'L_cross': 'E2 L_cross: map c, cross-city twin links only',
    'L_both': 'E2 L_both: map c, temporal + cross-city links (= baseline)',
    'N1_nuis_temp': 'E2 N1_nuis_temp: + 2-d city-level nuisance in x (not in f); temporal links only',
    'N1_nuis_both': 'E2 N1_nuis_both: + city-level nuisance; temporal + cross-city links',
    'N2_hc_temp': 'E2 N2_hc_temp: + city-constant structural feature in f (gamma 0.5); temporal links only -> its coefficient is not identified',
    'N2_hc_both': 'E2 N2_hc_both: + city-constant structural feature; cross-city links identify it',
    'F_5': 'E2 F_5: 5% of declared links are fake (hard arms use the eigen-gap rule)',
    'F_20': 'E2 F_20: 20% of declared links are fake',
    'S_25': 'E2 S_25: only 25% of the true links are declared',
    'S_5': 'E2 S_5: only 5% of the true links are declared',
    'A_02': 'E2 A_02: links only approximately stable (index gap sd 0.02 within-city SD)',
    'A_10': 'E2 A_10: links only approximately stable (index gap sd 0.10)',
    'D_rho08': 'E3 D_rho08: tract persistence of the inside error across dates, rho = 0.8',
    'D_sp005_ind': 'E3 D_sp005_ind: spatial random field (range 0.05, 50% of variance), independent across dates',
    'D_sp03_ind': 'E3 D_sp03_ind: spatial random field (range 0.3), independent across dates',
    'D_sp005_sh': 'E3 D_sp005_sh: spatial field (range 0.05) shared across dates',
    'D_sp03_sh': 'E3 D_sp03_sh: spatial field (range 0.3) shared across dates',
    'D_comb': 'E3 D_comb: rho = 0.8 + shared spatial field (range 0.3)',
    'X_i': 'E4 X_i: inside mean shift 0.6 A1 (A1 = skewed nuisance coordinate, clipped at 2.5 SD)',
    'X_ii': 'E4 X_ii: omitted variable correlated with A1 (corr 0.6), coefficient 0.7',
    'X_iii': 'E4 X_iii: noise scale and skew depend on A1; E[noise|x] = 0',
    'P_const': 'E5 P_const: 75% of the latent noise is a persistent tract part carried over with rho = 0.8 for every tract',
    'P_rdep': 'E5 P_rdep: persistence depends on latent wealth, rho(R) = 0.5 + 0.45 Phi(R) (richer tracts more persistent)',
    'P_reset': 'E5 P_reset: stable tracts carry the persistent part over (rho = 0.95); changing tracts redraw it (rho = 0)',
    'P_ferel': 'E5 P_ferel: literal fixed effect (rho = 1), but baselines are noisier for poorer tracts (uneven reliability)',
    'P_const_x': 'E5 P_const_x: as P_const, with half of the persistent part explained by x (xi = c f + xi_tilde)',
    'P_rdep_x': 'E5 P_rdep_x: as P_rdep, with half of the persistent part explained by x',
    'P_reset_x': 'E5 P_reset_x: as P_reset, with half of the persistent part explained by x',
    'P_ferel_x': 'E5 P_ferel_x: as P_ferel, with half of the persistent part explained by x',
}
METRIC_NOTES = {
    'r2': ('Panels: R\u00b2 of f = squared correlation of estimated and true index; R\u00b2 of W (dW) = out-of-sample R\u00b2 '
           'against OBSERVED held-out W (date-1 minus date-0 change of the same tract), within city-date (city), benchmark = '
           'the cell mean of training W. '
           'Dashed line per sample size = ceiling (the true E[W|x] as predictor). Heads: oracle = true map at the '
           'estimated index; CV B-spline = per city-date cubic B-spline, df by 5-fold CV (feasible head); native = FE/MM fit. '
           + R2_BELOW_NOTE),
    'rmse': ('Panels: RMSE of the standardised index (~ angular error, radians); RMSE of predicted W and dW against '
             'the TRUE conditional mean E[W|x]. Grey bar = asymptotic (pseudo-true) RMSE. Heads: oracle = true map '
             'at the estimated index; CV B-spline = per city-date cubic B-spline, df by 5-fold CV; native = FE/MM fit.'),
}


def footnote(scenario, arms, stat, width=200):
    import textwrap
    parts = [SCENARIO_NOTES.get(scenario, scenario),
             'Arms: ' + '; '.join(ARM_NOTES[a].split(' = ', 1)[0] + ' = ' + ARM_NOTES[a].split(' = ', 1)[1]
                                  for a in arms if a in ARM_NOTES),
             METRIC_NOTES[stat] + ' Whiskers = 95% Monte Carlo interval.']
    return '\n'.join(textwrap.fill(p, width) for p in parts)


def size_colors(sizes):
    k = len(sizes)
    ramp = SIZE_RAMP[k] if k in SIZE_RAMP else SIZE_RAMP[5][:1] if k == 1 else [SIZE_RAMP[5][int(round(i))] for i in np.linspace(0, 4, k)]
    return {s: ramp[min(i, len(ramp) - 1)] for i, s in enumerate(sizes)}


def _points(ax, x, y, xerr, stat, color, marker='o', s=30, face=None, edge=None, lw=0.8, elw=1.2):
    """Coefficient-plot points with 95% whiskers. On R2 axes, means below R2_FLOOR are drawn as a left-pointing
    marker at the floor without a whisker (how far below does not matter for the comparison)."""
    x, y, xerr = np.asarray(x, float), np.asarray(y, float), np.asarray(xerr, float)
    face = color if face is None else face
    edge = '#fcfcfb' if edge is None else edge
    unfilled = marker in ('x', '+', '|', '_')           # unfilled markers: colour via the face, no edge colour
    low = (x < R2_FLOOR) if stat == 'r2' else np.zeros(len(x), bool)
    ok = ~low & np.isfinite(x)
    if ok.any():
        ax.errorbar(x[ok], y[ok], xerr=xerr[ok], fmt='none', ecolor=color, elinewidth=elw, capsize=0, zorder=2)
        kw = dict(c=color) if unfilled else dict(facecolors=face, edgecolors=edge)
        ax.scatter(x[ok], y[ok], s=s, marker=marker, zorder=3, linewidths=lw * (2 if unfilled else 1), **kw)
    if low.any():
        ax.scatter(np.full(low.sum(), R2_FLOOR + 0.012), y[low], s=s * 1.3, marker='<', zorder=3, linewidths=lw,
                   facecolors=face, edgecolors=edge)


def _style_axis(ax, title, stat='rmse'):
    from matplotlib import ticker
    if stat == 'r2':
        ax.set_xlim(R2_FLOOR - 0.02, R2_TOP)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.axvline(0.0, color=MUTED, lw=0.6, zorder=0)
    if stat == 'rmse':
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(axis='x', which='both', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _pt_value(rec, target, stat):
    v = rec.get(PT_KEY[target])
    if v is None:
        return None
    if stat == 'rmse':
        return v
    return (1.0 - v ** 2 / 2.0) ** 2 if target == 'f' else None       # R2 pseudo-true only for the index


def scenario_figure(g, exp, scenario, pt_dir, out_dir, stat='r2'):
    """Coefficient plot: rows = arms, colour = sample size (C x n), marker = head, whiskers = +-1.96 MCSE.
    stat='r2': R2 (linear axis) with the per-size ceiling; stat='rmse': RMSE (log axis) with the pseudo-true bar."""
    g = g[(g['stat'] == stat) & (g['target'] != 'fpool')]
    if g.empty:
        return
    arms = [a for a in ARM_ROWS if a in set(g.arm)] + sorted(set(g.arm) - set(ARM_ROWS))
    sizes = sorted({(int(c), int(n)) for c, n in zip(g.C, g.n)}, key=lambda s: s[0] * s[1])
    col = size_colors(sizes)
    ns = len(sizes)
    dodge = np.linspace(0.28, -0.28, ns) if ns > 1 else np.zeros(1)
    ypos = {a: -i for i, a in enumerate(arms)}
    ptr = pt_lookup(exp, scenario, pt_dir) if pt_dir else None
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.42 * len(arms) + 3.2), sharey=True)
    for ax, (target, title) in zip(axes, PANELS[stat]):
        sub = g[g.target == target]
        heads = ['index'] if target == 'f' else HEADS_SHOWN
        for a in arms:
            if a in GROUP_BREAKS and a != arms[0]:
                ax.axhline(ypos[a] + 0.5, color=GRID, lw=1.0)
        if ptr:
            for a, rec in ptr['arms'].items():
                v = _pt_value(rec, target, stat)
                if a in ypos and v is not None:
                    ax.plot([v] * 2, [ypos[a] - 0.42, ypos[a] + 0.42], color=MUTED, lw=2.0,
                            solid_capstyle='round', zorder=1)
        if stat == 'r2' and target != 'f':
            for sz in sizes:
                c = sub[(sub['head'] == 'ceiling') & (sub.C == sz[0]) & (sub.n == sz[1])]
                if len(c):
                    ax.axvline(float(c['mean'].median()), color=col[sz], ls='--', lw=1.1, zorder=1)
        for j, sz in enumerate(sizes):
            d = sub[(sub.C == sz[0]) & (sub.n == sz[1])]
            for head in heads:
                h = d[d['head'] == head]
                if h.empty:
                    continue
                y = np.array([ypos[a] for a in h.arm]) + dodge[j]
                m = HEAD_MARK[head]
                _points(ax, h['mean'], y, 1.96 * h.mcse, stat, col[sz], marker=m['marker'], lw=1.3,
                        face=col[sz] if m['fill'] else '#fcfcfb', edge=col[sz])
        _style_axis(ax, title, stat)
        ax.set_xlabel('RMSE (log scale)' if stat == 'rmse' else 'R\u00b2', fontsize=9, color=MUTED)
    axes[0].set_yticks([ypos[a] for a in arms])
    axes[0].set_yticklabels(arms, fontsize=9, color=INK)
    axes[0].set_ylim(-len(arms) + 0.4, 0.6)
    from matplotlib.lines import Line2D
    h_size = [Line2D([], [], ls='none', marker='o', ms=7, color=col[s], label=f'{s[0]} cities x {s[1]}') for s in sizes]
    h_head = [Line2D([], [], ls='none', marker=HEAD_MARK[k]['marker'], ms=7, mec=MUTED, mew=1.3,
                     mfc=MUTED if HEAD_MARK[k]['fill'] else '#fcfcfb', label=HEAD_LABEL[k]) for k in HEADS_SHOWN]
    extra = []
    if ptr and stat == 'rmse':
        extra.append(Line2D([], [], color=MUTED, lw=2.0, label='asymptotic (pseudo-true)'))
    if stat == 'r2':
        extra.append(Line2D([], [], color=MUTED, lw=1.1, ls='--', label='ceiling (true E[W|x]), per size'))
    fig.suptitle(f'{exp} / {scenario}   mean over replications', fontsize=10, color=INK, y=0.995)
    fig.legend(handles=h_size + h_head + extra, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=len(h_size) + len(h_head) + len(extra), fontsize=8, frameon=False, handletextpad=0.3,
               columnspacing=1.2)
    fig.text(0.01, 0.005, footnote(scenario, arms, stat, width=250), fontsize=6.5, color=MUTED, va='bottom', ha='left')
    fig.tight_layout(rect=(0, 0.15, 1, 0.92))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'fig_{exp}_{scenario}.{ext}', dpi=150)
    plt.close(fig)


# overview across scenarios of one experiment: rows = scenarios, colour = arm (key arms only)
OVERVIEW_ARMS = ['RNC_S', 'RNC', 'SUR_S', 'FE', 'MM']
ARM_COLORS = dict(zip(OVERVIEW_ARMS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))   # validated
ARM_MARKS = dict(zip(OVERVIEW_ARMS, ['o', 's', '^', 'D', 'v']))                                  # secondary encoding


def overview_figure(agg, exp, out_dir, stat='r2', size=None):
    """One figure per experiment and size: rows = scenarios. Index panel: no head; W/dW panels: CV-spline head
    (the feasible one) for R2, oracle head for RMSE."""
    d = agg[(agg.exp == exp) & agg.arm.isin(OVERVIEW_ARMS) & (agg['stat'] == stat)]
    if d.empty:
        return
    wh = 'spline' if stat == 'r2' else 'oracle'
    sizes = sorted({(int(c), int(n)) for c, n in zip(d.C, d.n)}, key=lambda s: s[0] * s[1])
    for sz in ([size] if size else sizes):
        ds = d[(d.C == sz[0]) & (d.n == sz[1])]
        scen = [s for s in ex.SCENARIOS[exp] if s in set(ds.scenario)]
        if not scen:
            continue
        arms = [a for a in OVERVIEW_ARMS if a in set(ds.arm)]
        dodge = dict(zip(arms, np.linspace(0.3, -0.3, len(arms)) if len(arms) > 1 else [0.0]))
        ypos = {s: -i for i, s in enumerate(scen)}
        fig, axes = plt.subplots(1, 3, figsize=(14, 0.42 * len(scen) + 2.6), sharey=True)
        for ax, (target, title) in zip(axes, PANELS[stat]):
            head = 'index' if target == 'f' else wh
            sub = ds[(ds.target == target) & (ds['head'] == head)]
            for a in arms:
                h = sub[sub.arm == a]
                if h.empty:
                    continue
                y = np.array([ypos[s] for s in h.scenario]) + dodge[a]
                _points(ax, h['mean'], y, 1.96 * h.mcse, stat, ARM_COLORS[a], marker=ARM_MARKS[a])
            if stat == 'r2' and target != 'f':
                cz = ds[(ds.target == target) & (ds['head'] == 'ceiling')].groupby('scenario')['mean'].median()
                for s_, v in cz.items():
                    ax.plot([v, v], [ypos[s_] - 0.45, ypos[s_] + 0.45], color=MUTED, ls='--', lw=1.0, zorder=1)
            for s in scen[1:]:
                ax.axhline(ypos[s] + 0.5, color=GRID, lw=0.6)
            _style_axis(ax, title + ('' if target == 'f' else f' ({HEAD_LABEL[head]})'), stat)
            ax.set_xlabel('RMSE (log scale)' if stat == 'rmse' else 'R\u00b2', fontsize=9, color=MUTED)
        axes[0].set_yticks([ypos[s] for s in scen])
        axes[0].set_yticklabels(scen, fontsize=9, color=INK)
        axes[0].set_ylim(-len(scen) + 0.4, 0.6)
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], ls='none', marker=ARM_MARKS[a], ms=7, color=ARM_COLORS[a], label=a) for a in arms]
        if stat == 'r2':
            hd.append(Line2D([], [], color=MUTED, ls='--', lw=1.0, label='ceiling (true E[W|x])'))
        fig.suptitle(f'{exp}: all scenarios at {sz[0]} cities x {sz[1]} tracts   (whiskers = 95% MC interval)',
                     fontsize=10, color=INK, y=0.995)
        fig.legend(handles=hd, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=len(hd), fontsize=8, frameon=False)
        import textwrap
        note = ('Scenarios: see paper/DESIGN.md. Arms: ' + '; '.join(ARM_NOTES[a] for a in arms) + '. ' + METRIC_NOTES[stat])
        fig.text(0.01, 0.005, textwrap.fill(note, 250), fontsize=6.5, color=MUTED, va='bottom', ha='left')
        fig.tight_layout(rect=(0, 0.13, 1, 0.92))
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'pdf'):
            fig.savefig(out_dir / f'fig_overview_{exp}_{sz[0]}x{sz[1]}.{ext}', dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# E5 (and E3 D_rho08): baseline updating
# ---------------------------------------------------------------------------
UPD_ROWS = ['survey', 'image', 'imgchg', 'kconst', 'kR', 'oracle']
UPD_LABEL = {'survey': 'survey only (baseline rank)', 'image': 'image head only (kappa = 0)',
             'imgchg': 'baseline + image change (kappa = 1)', 'kconst': 'constant kappa',
             'kR': 'kappa(R) linear', 'oracle': 'oracle E[W1 | x, W0]'}
UPD_PANELS = [('r2', 'dWo', 'R\u00b2 of observed dW: all tracts'),
              ('r2', 'dWoS', 'R\u00b2 of observed dW: stable tracts'),
              ('r2', 'dWoC', 'R\u00b2 of observed dW: changing tracts'),
              ('r2', 'dWsC', 'R\u00b2 of the structural change (changing tracts)')]
MARGINAL_LABEL = 'best with the marginal curve E[W1 | index] (structural panel)'
UPD_NOTE = ('Predictors of the date-1 outcome for held-out tracts whose baseline W0 is observed (dW = W1 - W0): '
            'survey only = baseline rank mapped to the date-1 city distribution; image head = CV-spline head on '
            'the date-1 index; baseline + image change = W0 + h1(R1) - h0(R0); constant kappa / kappa(R) = image '
            'head + carry-over weight x baseline residual, weight estimated on training tracts labelled at both '
            'dates; oracle = true E[W1 | x, W0]. R\u00b2 of observed dW is within city and within group (benchmark: '
            'the city mean of the same group in training). Structural change = the part of E[dW | x, W0] caused by '
            'the index change (0 for stable tracts); a predictor\'s structural part = h1(R1) - h1(R0). The oracle '
            'scores 1 on it by construction (it is the target). Dashed mark in that panel = the best any predictor built on '
            'the marginal curve E[W1 | index] can do (true map and index, no estimation error): its gap to 1 is the part '
            'of the structural change that depends on the tract\'s baseline residual under a nonlinear map; the gap '
            'below it is estimation error (first step + head). First step: RNC_S. ' + R2_BELOW_NOTE)


def update_figure(g, exp, scenario, out_dir, arm='RNC_S'):
    mg = g[(g.arm == arm) & (g['head'] == 'marginal') & (g.target == 'dWsC') & (g['stat'] == 'r2')]
    g = g[(g.arm == arm) & g['head'].isin(UPD_ROWS)]
    if g.empty or not (g.target == 'dWo').any():
        return
    sizes = sorted({(int(c), int(n)) for c, n in zip(g.C, g.n)}, key=lambda s: s[0] * s[1])
    col = size_colors(sizes)
    rows = [h for h in UPD_ROWS if h in set(g['head'])]
    ypos = {h: -i for i, h in enumerate(rows)}
    dodge = np.linspace(0.25, -0.25, len(sizes)) if len(sizes) > 1 else np.zeros(1)
    fig, axes = plt.subplots(1, len(UPD_PANELS), figsize=(18, 0.45 * len(rows) + 3.0), sharey=True)
    for ax, (stat, target, title) in zip(axes, UPD_PANELS):
        sub = g[(g['stat'] == stat) & (g.target == target)]
        for j, sz in enumerate(sizes):
            d = sub[(sub.C == sz[0]) & (sub.n == sz[1])]
            if d.empty:
                continue
            y = np.array([ypos[h] for h in d['head']]) + dodge[j]
            _points(ax, d['mean'], y, 1.96 * d.mcse, stat, col[sz])
            m = mg[(mg.C == sz[0]) & (mg.n == sz[1])]
            if target == 'dWsC' and len(m):
                ax.axvline(float(m['mean'].median()), color=col[sz], ls='--', lw=1.1, zorder=1)
        _style_axis(ax, title, 'r2')
        ax.set_xlabel('R\u00b2', fontsize=9, color=MUTED)
    axes[0].set_yticks([ypos[h] for h in rows])
    axes[0].set_yticklabels([UPD_LABEL[h] for h in rows], fontsize=9, color=INK)
    axes[0].set_ylim(-len(rows) + 0.4, 0.6)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls='none', marker='o', ms=7, color=col[s], label=f'{s[0]} cities x {s[1]}') for s in sizes]
    if len(mg):
        hs.append(Line2D([], [], color=MUTED, ls='--', lw=1.1, label=MARGINAL_LABEL))
    fig.suptitle(f'{exp} / {scenario}: baseline updating ({arm} first step)', fontsize=10, color=INK, y=0.995)
    fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=len(hs), fontsize=8, frameon=False)
    import textwrap
    note = SCENARIO_NOTES.get(scenario, scenario) + '\n' + textwrap.fill(UPD_NOTE + ' Whiskers = 95% Monte Carlo interval.', 250)
    fig.text(0.01, 0.005, note, fontsize=6.5, color=MUTED, va='bottom', ha='left')
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'fig_update_{exp}_{scenario}.{ext}', dpi=150)
    plt.close(fig)


UPD_COLORS = dict(zip(['survey', 'image', 'imgchg', 'kconst', 'kR', 'oracle'],
                      ['#eda100', '#eb6834', '#e87ba4', '#1baf7a', '#2a78d6', '#52514e']))
UPD_MARKS = dict(zip(['survey', 'image', 'imgchg', 'kconst', 'kR', 'oracle'], ['v', 's', 'D', '^', 'o', 'x']))


def update_overview(agg, exp, out_dir, arm='RNC_S'):
    d = agg[(agg.exp == exp) & (agg.arm == arm) & agg['head'].isin(UPD_ROWS)]
    if d.empty or not (d.target == 'dWo').any():
        return
    sizes = sorted({(int(c), int(n)) for c, n in zip(d.C, d.n)}, key=lambda s: s[0] * s[1])
    for sz in sizes:
        ds = d[(d.C == sz[0]) & (d.n == sz[1])]
        scen = [s for s in ex.SCENARIOS[exp] if s in set(ds.scenario) and (ds[ds.scenario == s].target == 'dWo').any()]
        if not scen:
            continue
        rows = [h for h in UPD_ROWS if h in set(ds['head'])]
        dodge = dict(zip(rows, np.linspace(0.32, -0.32, len(rows))))
        ypos = {s: -i for i, s in enumerate(scen)}
        fig, axes = plt.subplots(1, len(UPD_PANELS), figsize=(18, 0.5 * len(scen) + 2.8), sharey=True)
        for ax, (stat, target, title) in zip(axes, UPD_PANELS):
            sub = ds[(ds['stat'] == stat) & (ds.target == target)]
            for h in rows:
                hh = sub[sub['head'] == h]
                hh = hh[hh.scenario.isin(scen)]
                if hh.empty:
                    continue
                y = np.array([ypos[s] for s in hh.scenario]) + dodge[h]
                _points(ax, hh['mean'], y, 1.96 * hh.mcse, stat, UPD_COLORS[h], marker=UPD_MARKS[h], s=28,
                        edge=UPD_COLORS[h], elw=1.1)
            if target == 'dWsC':                         # attainable with the marginal curve E[W1 | index]
                mg = agg[(agg.exp == exp) & (agg.arm == arm) & (agg['head'] == 'marginal') & (agg.target == 'dWsC')
                         & (agg.C == sz[0]) & (agg.n == sz[1]) & (agg['stat'] == 'r2')]
                for s_, v in mg.groupby('scenario')['mean'].median().items():
                    if s_ in ypos:
                        ax.plot([v, v], [ypos[s_] - 0.45, ypos[s_] + 0.45], color=MUTED, ls='--', lw=1.0, zorder=1)
            for s_ in scen[1:]:
                ax.axhline(ypos[s_] + 0.5, color=GRID, lw=0.6)
            _style_axis(ax, title, 'r2')
            ax.set_xlabel('R\u00b2' if stat == 'r2' else 'RMSE', fontsize=9, color=MUTED)
        axes[0].set_yticks([ypos[s] for s in scen])
        axes[0].set_yticklabels(scen, fontsize=9, color=INK)
        axes[0].set_ylim(-len(scen) + 0.4, 0.6)
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], ls='none', marker=UPD_MARKS[h], ms=7, color=UPD_COLORS[h], label=UPD_LABEL[h]) for h in rows]
        hd.append(Line2D([], [], color=MUTED, ls='--', lw=1.0, label=MARGINAL_LABEL))
        fig.suptitle(f'{exp}: baseline updating, all scenarios at {sz[0]} cities x {sz[1]} tracts ({arm} first step)',
                     fontsize=10, color=INK, y=0.995)
        fig.legend(handles=hd, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=3, fontsize=8, frameon=False)
        import textwrap
        fig.text(0.01, 0.005, textwrap.fill('Scenarios: see paper/DESIGN.md. ' + UPD_NOTE, 250), fontsize=6.5,
                 color=MUTED, va='bottom', ha='left')
        fig.tight_layout(rect=(0, 0.1, 1, 0.9))
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'pdf'):
            fig.savefig(out_dir / f'fig_update_overview_{exp}_{sz[0]}x{sz[1]}.{ext}', dpi=150)
        plt.close(fig)


DEC_PARTS = [('map', 'city map moving', '#2a78d6'), ('struct', 'structural (index) change', '#eb6834'),
             ('resid', 'residual dynamics', '#1baf7a'), ('noise', 'unpredictable', '#c9c8c3')]


def decomposition_figure(agg, exp, out_dir, arm='RNC_S'):
    """Where the within-city variance of observed dW comes from (oracle decomposition), all tracts and
    changing tracts, one stacked bar per scenario and group, at the largest size."""
    d = agg[(agg.exp == exp) & (agg.arm == arm) & agg['stat'].isin(['vdec', 'vdecC'])]
    if d.empty:
        return
    C, n = max({(int(c), int(m)) for c, m in zip(d.C, d.n)}, key=lambda s: s[0] * s[1])
    d = d[(d.C == C) & (d.n == n)]
    scen = [s for s in ex.SCENARIOS[exp] if s in set(d.scenario)]
    labels, rows = [], []
    for sc in scen:
        for stat, grp in (('vdec', 'all tracts'), ('vdecC', 'changing tracts')):
            v = d[(d.scenario == sc) & (d['stat'] == stat)].set_index('target')['mean']
            rows.append([float(v.get(p, np.nan)) for p, _, _ in DEC_PARTS])
            labels.append(f'{sc}: {grp}')
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(rows) + 2.2))
    y = -np.arange(len(rows))
    left = np.zeros(len(rows))
    for j, (p, lab, colr) in enumerate(DEC_PARTS):
        ax.barh(y, rows[:, j], left=left, height=0.7, color=colr, edgecolor='#fcfcfb', linewidth=2, label=lab)
        left += np.nan_to_num(rows[:, j])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    _style_axis(ax, f'{exp}: share of the within-city variance of observed dW, by source ({C} cities x {n})', 'r2')
    ax.set_xlabel('share of the within-city variance of observed dW (sums to 1)', fontsize=8, color=MUTED)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=8, frameon=False)
    import textwrap
    fig.text(0.01, 0.005, textwrap.fill(
        'Oracle decomposition of each tract\'s expected change E[dW | x, W0] = map + structural + residual: map = '
        'psi_1 - psi_0 at the tract\'s structural position (the city map moving: the same function of the index for '
        'every tract in the city); structural = change caused by the index change (0 for stable tracts); residual = '
        'expected movement of the tract residual (mean reversion, persistence). Shares = Shapley split of the '
        'within-city R\u00b2 of observed dW among the three; unpredictable = 1 - oracle R\u00b2.', 200),
        fontsize=6.5, color=MUTED, va='bottom', ha='left')
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'fig_dW_sources_{exp}.{ext}', dpi=150)
    plt.close(fig)


def omega_figure(om, scenario, C, n, lam, out_dir):
    d = om[(om.scenario == scenario) & (om.C == C) & (om.n == n)]
    if d.empty:
        return
    curve = d[d.arm != 'RNC_SEL'].sort_values('omega')
    if len(curve) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4))
    x = np.arange(len(curve))
    ax.errorbar(x, curve.rmse_f, 1.96 * curve.rmse_f_mcse, marker='o', color='#D55E00', capsize=3)
    labs = []
    for w in curve.omega:
        labs.append('0' if w == 0 else 'hard' if w == 1 else f'{w}\nlam={lam.get(str(w), np.nan):.3g}')
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
    s = d[d.arm == 'RNC_SEL']
    if len(s):
        ax.axhline(float(s.rmse_f.iloc[0]), color='#8C564B', ls='--', label=f'selector (mean omega {float(s.omega_sel_mean.iloc[0]):.2f})')
        ax.legend(fontsize=8)
    ax.set_yscale('log'); ax.set_xlabel('omega (calibrated fixed lambda marked)'); ax.set_ylabel('RMSE of index f')
    ax.set_title(f'{scenario}  {C}x{n}', fontsize=10); ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'fig_omega_{scenario}_{C}x{n}.{ext}', dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--pseudotrue', default=str(HERE / 'pseudotrue'))
    ap.add_argument('--policy', default=None)
    ap.add_argument('--stat', default='r2,rmse', help='r2 (figures/) and/or rmse (figures/rmse/)')
    a = ap.parse_args(argv)
    out = Path(a.out or Path(a.results.split(',')[0]) / 'figures')
    df = sm.load(a.results)
    if df.empty:
        print('no results')
        return
    agg = sm.aggregate(sm.tidy(df))
    for stat in a.stat.split(','):
        od = out if stat == 'r2' else out / stat
        for (exp, scen), g in agg.groupby(['exp', 'scenario']):
            scenario_figure(g, exp, scen, a.pseudotrue, od, stat)
        for exp in sorted(set(agg.exp)):
            overview_figure(agg, exp, od, stat)
    for (exp, scen), g in agg.groupby(['exp', 'scenario']):
        update_figure(g, exp, scen, out)
    for exp in sorted(set(agg.exp)):
        update_overview(agg, exp, out)
        decomposition_figure(agg, exp, out)
    om = sm.omega_table(df)
    pol = json.loads(Path(a.policy or HERE / 'calibration.json').read_text()) if Path(a.policy or HERE / 'calibration.json').exists() else {}
    for (scen, C, n), _ in om.groupby(['scenario', 'C', 'n']):
        omega_figure(om, scen, C, n, pol.get('lambda', {}), out)
    print('figures ->', out)


if __name__ == '__main__':
    main()
