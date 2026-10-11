"""Figures 1 to 3 of the paper, written to ../figures/ (about a minute; the arc-length curve dominates).
  fig1_declinations.png  asymptotic declinations of the twelve Earth flybys and of Europa Clipper (Section 2)
  fig2_tracking.png      (a) Deep Space Network elevations through perigee; (b) 3-sigma detection limit against the
                         half-length of the tracking arc (Section 4)
  fig3_test.png          (a) predictions against the 3-sigma limits of Table 3; (b) conventional effects of Table 4"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import astg
from predictions import predict
from covariance import propagate, ecef, rotation, DSN, analyse, arc_dependence
from perturbations import shadow_times, budget
from outcomes import PESSIMISTIC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'dejavusans'})
BLUE, RED, GREEN, GREY, ORANGE = '#1f4e79', '#b22222', '#2e7d32', '0.45', '#c66a00'
ORDER = ['Galileo I', 'Galileo II', 'NEAR', 'Cassini', 'Rosetta I', 'EPOXI III', 'MESSENGER', 'Rosetta II', 'EPOXI I',
         'Rosetta III', 'EPOXI II', 'Juno']
OFFSETS = {'Galileo I': (-7, -3, 'right'), 'Galileo II': (-6, 7, 'left'), 'NEAR': (7, -3, 'left'), 'Cassini': (6, 5, 'left'),
           'Rosetta I': (7, -3, 'left'), 'EPOXI III': (-14, -15, 'left'), 'MESSENGER': (7, -3, 'left'), 'Rosetta II': (-7, 2, 'right'),
           'EPOXI I': (6, -11, 'left'), 'Rosetta III': (7, -9, 'left'), 'EPOXI II': (7, -3, 'left'), 'Juno': (7, -3, 'left')}


def figure1(p):
    fl = astg.load_flybys()
    fig, ax = plt.subplots(figsize=(2300 / 400, 2250 / 400), dpi=400)
    L = 80
    ax.add_patch(Rectangle((-L, -L), L, L, color='#dfe8f1', zorder=0))
    ax.text(-77, -50, 'both declinations\nnegative: the law\napplies under the\nselection rule', fontsize=8.5, color=BLUE, va='bottom')
    xx = np.linspace(-L, L, 2)
    ax.plot(xx, xx, ':', color='0.55', lw=0.9); ax.plot(xx, -xx, ':', color='0.55', lw=0.9)
    ax.text(56, 61, r'$\Delta\cos\delta=0$', fontsize=8.5, color='0.4', rotation=45, ha='center')
    ax.axhline(0, color='0.75', lw=0.7); ax.axvline(0, color='0.75', lw=0.7)
    for n in ORDER:
        di, do = fl[n]['delta_in_deg'], fl[n]['delta_out_deg']
        if fl[n]['class'] != 'null':
            ax.plot(di, do, 'o', color=RED, ms=7, mec='k', mew=0.5, zorder=3)
        else:
            ax.plot(di, do, 'o', mfc='white', mec=BLUE, mew=1.4, ms=7, zorder=3)
        ox, oy, ha = OFFSETS[n]
        ax.annotate(n, (di, do), xytext=(ox, oy), textcoords='offset points', fontsize=8.5, ha=ha)
    clip = (p['g']['din'], p['g']['dout'])
    ax.plot(*clip, '*', color='k', ms=15, zorder=4)
    ax.annotate('Europa Clipper\n3 Dec 2026', clip, xytext=(10, 2), textcoords='offset points', fontsize=9, fontweight='bold')
    ax.plot([], [], 'o', color=RED, mec='k', mew=0.5, ms=7, label='anomalous')
    ax.plot([], [], 'o', mfc='white', mec=BLUE, mew=1.4, ms=7, label='null')
    ax.plot([], [], '*', color='k', ms=12, label='Europa Clipper (this work)')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
    ax.set_xlabel(r'$\delta_{\rm in}$ (deg)'); ax.set_ylabel(r'$\delta_{\rm out}$ (deg)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1_declinations.png'), dpi=400); plt.close(fig)


def figure2(p, sh, arcs):
    fw, bw = propagate(5.0)
    sta = {k: ecef(*v) for k, v in DSN.items()}
    ts = np.arange(-12 * 3600, 12 * 3600 + 1, 60.0)
    el = {k: [] for k in DSN}
    for t in ts:
        r = (fw if t >= 0 else bw).sol(t)[:3]; R = rotation(t)
        for k, (pe, up) in sta.items():
            d = r - R @ pe; el[k].append(np.degrees(np.arcsin(d @ (R @ up) / np.linalg.norm(d))))
    fig, axs = plt.subplots(1, 2, figsize=(3840 / 400, 1700 / 400), dpi=400, gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = axs[0]
    cols = {'Goldstone DSS-25': ORANGE, 'Canberra DSS-34': GREEN, 'Madrid DSS-54': BLUE}
    for k in DSN:
        a.plot(ts / 3600, el[k], color=cols[k], lw=1.4, label=k.split()[0])
    a.axhline(10, color='k', ls='--', lw=0.8); a.text(-11.6, 12, '10° mask', fontsize=8)
    a.axvspan(sh['pen_in'] / 3600, sh['pen_out'] / 3600, color='0.25', alpha=0.25, lw=0); a.text(0.6, 75, 'Earth\nshadow', fontsize=8, ha='left')
    a.axvline(0, color=RED, lw=1.0); a.text(-0.3, 75, 'perigee', fontsize=8, ha='right', color=RED)
    a.set_ylim(0, 90); a.set_xlim(-12, 12); a.set_xticks(range(-12, 13, 4))
    a.set_xlabel('time from perigee (h)'); a.set_ylabel('elevation (deg)')
    a.set_title('(a) Deep Space Network visibility', fontsize=11); a.legend(frameon=False, fontsize=8, loc='upper left')
    arcs = np.array(arcs)
    b = axs[1]
    for nz, al in ((0.2, 0.55), (0.1, 1.0), (0.05, 0.55)):
        b.plot(arcs[:, 0], 3 * arcs[:, 1] * nz / 0.1, '-', color=BLUE, lw=2.0 if nz == 0.1 else 1.1, alpha=al, label=f'noise {nz:g} mm s$^{{-1}}$')
    b.plot(arcs[:, 0], 3 * arcs[:, 2], '--', color=BLUE, lw=1.4, label='noise 0.1 mm s$^{-1}$, SRP estimated')
    for v, lab, col in ((p['anderson_p'], 'Anderson\nrelation', GREY), (p['law4'], 'ASTG law', RED),
                        (p['single']['Rosetta I'], 'lowest single-\nencounter\ncoupling', RED)):
        b.axhline(v, xmax=(7.0 - 0.4) / (10.0 - 0.4), color=col, ls=':', lw=1.3)
        b.text(7.2, v, lab, fontsize=8, color=col, va='center')
    b.axvline(3, ymin=0.25, color='0.6', lw=0.8)
    b.set_yscale('log'); b.set_ylim(0.001, 0.6); b.set_xlim(0.4, 10.0); b.set_xticks([0.5, 1, 2, 3, 4, 5, 7])
    b.set_xticklabels(['0.5', '1', '2', '3', '4', '5', '7'])
    b.set_xlabel('tracking arc, days either side of perigee'); b.set_ylabel(r'3$\sigma$ detection limit (mm s$^{-1}$)')
    b.set_title('(b) detection limit against arc length', fontsize=11)
    b.legend(frameon=False, fontsize=7.5, loc='lower left', handlelength=2.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_tracking.png'), dpi=400); plt.close(fig)


def figure3(p, s_base, s_pess, bud):
    low, mid, high = (p['single'][k] for k in ('Rosetta I', 'NEAR', 'Galileo I'))
    fig, axs = plt.subplots(1, 2, figsize=(3840 / 400, 1600 / 400), dpi=400, gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = axs[0]
    rows = [('law + selection rule', 0.0, None, None, 'k'),
            ('law, four encounters', p['law4'], p['s_law4'], None, RED),
            ('law, five-encounter refit', p['law5'], p['s_law5'], None, RED),
            ('law, two angular terms', p['two_term'], p['s_two_term'], None, RED),
            ('law, single-encounter couplings', None, None, (low, high), RED),
            ('Anderson relation', p['anderson_p'], None, None, GREY)]
    y = np.arange(len(rows))[::-1]
    for (lab, v, s, rng, col), yy in zip(rows, y):
        if rng is not None:
            a.plot(rng, [yy, yy], color=col, lw=6, alpha=0.35, solid_capstyle='butt'); a.plot(mid, yy, 'o', color=col, mec='k', mew=0.5, ms=6)
        elif s is not None:
            a.errorbar(v, yy, xerr=s, fmt='o', color=col, mec='k', mew=0.5, ms=7, capsize=3)
        else:
            a.plot(v, yy, 'D' if lab.startswith('law +') else 'o', color=col, mec='k', mew=0.5, ms=7)
    a.axvspan(-3 * s_base, 3 * s_base, color=BLUE, alpha=0.18, lw=0); a.axvspan(-3 * s_pess, 3 * s_pess, color=BLUE, alpha=0.08, lw=0)
    a.text(0.0, y[0] + 0.62, '3σ, baseline', fontsize=7.5, ha='center', color=BLUE)
    a.text(-0.040, y[-1] - 0.45, '3σ, pessimistic', fontsize=7.5, ha='left', color=BLUE)
    a.axvline(0, color='k', lw=0.6)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=9); a.set_ylim(-0.8, len(rows) - 0.3)
    a.set_xlim(-0.05, 0.33); a.set_xlabel(r'predicted $\Delta V_p$ (mm s$^{-1}$)'); a.set_title('(a) predictions and detection limits', fontsize=11)
    b = axs[1]
    eff = [('Earth shadow', abs(bud['shadow']), bud['after']['shadow']), ('array thermal recoil', bud['thermal_full'], bud['after']['thermal']),
           ('Earth albedo', abs(bud['albedo']['dvp']), bud['after']['albedo']), ('Earth infrared', abs(bud['infrared']['dvp']), bud['after']['infrared']),
           ('atmospheric drag', abs(bud['drag']), bud['after']['drag'])]
    yv = np.arange(len(eff))[::-1]
    b.barh(yv + 0.17, [e[1] for e in eff], color='#b9cde3', edgecolor='0.3', lw=0.5, height=0.32, label='unmodelled')
    b.barh(yv - 0.17, [e[2] for e in eff], color='#1f4e79', edgecolor='0.2', lw=0.5, height=0.32, label='after modelling')
    b.axvline(p['law4'], color=RED, ls='--', lw=1.1); b.text(0.058, yv[-1] - 0.25, 'ASTG\nlaw', color=RED, fontsize=8, va='center')
    b.axvline(s_base, color=BLUE, ls=':', lw=1.2); b.text(s_base * 0.93, yv[-1] - 0.45, '1σ', color=BLUE, fontsize=8, ha='right')
    b.legend(frameon=False, fontsize=8, loc='upper right', bbox_to_anchor=(1.02, 1.04))
    b.set_xscale('log'); b.set_xlim(5e-5, 0.6); b.set_yticks(yv); b.set_yticklabels([e[0] for e in eff], fontsize=9); b.set_ylim(-0.8, len(eff) - 0.2)
    b.set_xlabel(r'$|\Delta V_p|$ (mm s$^{-1}$)'); b.set_title('(b) conventional effects', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_test.png'), dpi=400); plt.close(fig)


if __name__ == '__main__':
    from thermal import thermal_cases
    os.makedirs(OUT, exist_ok=True)
    p = predict()
    figure1(p)
    figure2(p, shadow_times(), arc_dependence())
    th = thermal_cases()
    bud = budget(thermal_lag_max=max(abs(c['lag']) for c in th)); bud['thermal_full'] = max(c['full'] for c in th)
    figure3(p, analyse()[0], analyse(**PESSIMISTIC)[0], bud)
    print('wrote', os.path.normpath(os.path.join(OUT, 'fig1_declinations.png')), ', fig2_tracking.png and fig3_test.png')
