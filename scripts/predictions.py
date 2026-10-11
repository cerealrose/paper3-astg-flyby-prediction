"""Predictions for the Europa Clipper encounter (Sections 2.2 and 3, Table 2, Eqs. 4 to 6, Appendix A).

  * ASTG law, Eq. (2), with the four-encounter coupling of the first companion paper and the five-encounter refit
    of the second (Galileo II at its drag-corrected residual, +1.7628 mm/s, and Cassini at -0.42 mm/s).
  * Two-term angular law of the second companion paper: coefficients on dcos and dsin fitted to the four anomalous
    encounters, with their covariance.
  * Single-encounter couplings of Galileo I, NEAR and Rosetta I, the three measured to better than 8%.
  * Selection rule, Eq. (5): zero unless both declinations are negative.
  * Anderson relation, Eq. (1), and the ratio of Eq. (6).
Encounter data for the couplings are read from data/earth_flybys.csv."""
import numpy as np
import astg
import clipper as cl

GALILEO_II_RESIDUAL = 1.7628        # mm/s, drag-corrected (second companion paper)
CASSINI_REVISED = -0.42             # mm/s (Jouannic et al. 2015)
SINGLE = ('Galileo I', 'NEAR', 'Rosetta I')


def couplings(fl=None):
    fl = fl or astg.load_flybys()
    four = astg.anomalous_fit(fl)
    five = astg.anomalous_fit(fl, dvp_override={'Cassini': CASSINI_REVISED, 'Galileo II': GALILEO_II_RESIDUAL},
                              names=astg.ANOMALOUS + ['Galileo II'])
    return four, five


def single_couplings(fl=None):
    fl = fl or astg.load_flybys()
    out = {}
    for k in ('Galileo I', 'NEAR', 'Cassini', 'Rosetta I'):
        d = astg.derived(fl[k])
        out[k] = (fl[k]['dVp_mm_s'] * 1e-3 / d['vp'] / d['X'], fl[k]['sigma_dVp_mm_s'] * 1e-3 / d['vp'] / abs(d['X']),
                  fl[k]['sigma_dVp_mm_s'] / abs(fl[k]['dVp_mm_s']))
    return out


def two_term(fl=None):
    """Coefficients on dcos and dsin from the four anomalous encounters, with covariance (two degrees of freedom)."""
    fl = fl or astg.load_flybys()
    X, W, Y = [], [], []
    for k in astg.ANOMALOUS:
        d = astg.derived(fl[k]); f = astg.RG / (2 * d['semilatus'])
        X.append(f * d['dcos']); W.append(f * d['dsin']); Y.append(fl[k]['dVp_mm_s'] * 1e-3 / d['vp'])
    A = np.column_stack([X, W]); Y = np.array(Y)
    co = np.linalg.lstsq(A, Y, rcond=None)[0]; r = Y - A @ co
    cov = r @ r / (len(Y) - 2) * np.linalg.inv(A.T @ A)
    return co, cov


def predict(fl=None):
    fl = fl or astg.load_flybys()
    g = cl.encounter('date')
    rp = g['rp'] * 1e3; vinf = g['vinf'] * 1e3; vp = np.sqrt(vinf**2 + 2 * astg.GM / rp)
    eps = 1 + rp * vinf**2 / astg.GM
    x = astg.RG * g['dcos'] / (2 * rp * (1 + eps))
    four, five = couplings(fl)
    law4 = four['lam'] * x * vp * 1e3; law5 = five['lam'] * x * vp * 1e3
    co, cov = two_term(fl)
    grad = np.array([g['dcos'], g['dsin']]) * astg.RG / (2 * rp * (1 + eps)) * vp * 1e3
    sc = single_couplings(fl)
    single = {k: sc[k][0] * x * vp * 1e3 for k in SINGLE}
    rule = law4 if (g['din'] < 0 and g['dout'] < 0) else 0.0
    and_inf = astg.KAPPA_A * g['dcos'] * vinf * 1e3; and_p = and_inf * vinf / vp
    return dict(g=g, rp=rp, vinf=vinf, vp=vp, eps=eps, x=x, four=four, five=five,
                law4=law4, s_law4=law4 * four['se'] / four['lam'], law5=law5, s_law5=law5 * five['se'] / five['lam'],
                energy=four['lam'] * x * vp**2, two_term=grad @ co, s_two_term=np.sqrt(grad @ cov @ grad),
                two_term_coef=co, two_term_se=np.sqrt(np.diag(cov)), two_term_corr=cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
                single=single, single_couplings=sc, rule=rule, anderson_inf=and_inf, anderson_p=and_p, to_inf=vp / vinf)


def sensitivities(p=None):
    """Section 2.2: derivatives of Eq. (2) (four-encounter coupling) and of the Anderson relation at perigee."""
    p = p or predict(); g = p['g']; lam = p['four']['lam']
    hp, din, dout, vinf = g['hp'], g['din'], g['dout'], g['vinf']
    base = cl.law(lam, hp, din, dout, vinf)
    d_h = (cl.law(lam, hp + 1, din, dout, vinf) - cl.law(lam, hp - 1, din, dout, vinf)) / 2
    d_in = (cl.law(lam, hp, din + 0.005, dout, vinf) - cl.law(lam, hp, din - 0.005, dout, vinf)) / 0.01
    d_out = (cl.law(lam, hp, din, dout + 0.005, vinf) - cl.law(lam, hp, din, dout - 0.005, vinf)) / 0.01
    vp = p['vp'] / 1e3
    a_in = -astg.KAPPA_A * np.sin(din * cl.D2R) * cl.D2R * vinf**2 / vp * 1e6
    a_out = astg.KAPPA_A * np.sin(dout * cl.D2R) * cl.D2R * vinf**2 / vp * 1e6
    return dict(base=base, d_h=d_h, d_in=d_in, d_out=d_out, pct_10km=abs(10 * d_h / base) * 100,
                pct_001deg=max(abs(d_in), abs(d_out)) * 0.01 / base * 100, anderson_d_in=a_in, anderson_d_out=a_out)


def ratio_eq6(rp_m, vinf_ms, lam):
    return lam * astg.GM**2 / (astg.KAPPA_A * astg.C0**2 * rp_m**2 * vinf_ms**2)


def appendix_a():
    r = np.linalg.norm(cl.R_DATE); v = np.linalg.norm(cl.V_DATE); g = cl.asymptotes(cl.R_DATE, cl.V_DATE)
    ci, co = np.cos(g['din'] * cl.D2R), np.cos(g['dout'] * cl.D2R)
    semilatus = g['rp'] * 1e3 * (1 + g['e'])
    dvp = v * 1e3 * 3902 * 8.870e-3 * (ci - co) / (2 * semilatus) * 1e3
    return dict(r_vec=cl.R_DATE, v_vec=cl.V_DATE, r=r, v=v, E=g['E'], vinf=np.sqrt(2 * g['E']), e=g['e'], rp=g['rp'],
                semilatus=semilatus, sz_in=g['s_in'][2], sz_out=g['s_out'][2], din=g['din'], dout=g['dout'],
                cos_in=ci, cos_out=co, dcos=ci - co, dvp=dvp)


if __name__ == '__main__':
    fl = astg.load_flybys(); p = predict(fl); g = p['g']
    print(f"Clipper: rp {p['rp'] / 1e3:.3f} km, Vinf {p['vinf']:.2f} m/s, Vp {p['vp']:.2f} m/s, eps {p['eps']:.5f}, "
          f"dcos {g['dcos']:+.6f}, dsin {g['dsin']:+.6f}, X {p['x']:.5e}")
    s = sensitivities(p)
    print(f"\nSection 2.2: d/dh_p {s['d_h']:+.2e} mm/s per km ({s['pct_10km']:.2f}% for 10 km); d/d delta_in {s['d_in']:+.4f}, "
          f"d/d delta_out {s['d_out']:+.4f} mm/s per deg ({s['pct_001deg']:.2f}% for 0.01 deg)")
    print(f"  Anderson relation: {s['anderson_d_in']:+.4f} and {s['anderson_d_out']:+.4f} mm/s per deg")
    print(f"\nEq. (4): {p['law4']:+.4f} +- {p['s_law4']:.4f} mm/s at perigee, {p['law4'] * p['to_inf']:+.4f} +- {p['s_law4'] * p['to_inf']:.4f} at infinity;"
          f" energy Vp dVp = {p['energy']:.3f} J/kg")
    print(f"five-encounter coupling ({p['five']['lam'] / 1e3:.3f} +- {p['five']['se'] / 1e3:.3f} e3): {p['law5']:+.4f} +- {p['s_law5']:.4f}"
          f" mm/s; at infinity {p['law5'] * p['to_inf']:+.4f} +- {p['s_law5'] * p['to_inf']:.4f}")
    co, se = p['two_term_coef'], p['two_term_se']
    print(f"two-term law (lambda_a {co[0] / 1e3:.2f} +- {se[0] / 1e3:.2f}, lambda_b {co[1] / 1e3:.2f} +- {se[1] / 1e3:.2f} e3, "
          f"correlation {p['two_term_corr']:+.2f}): {p['two_term']:+.4f} +- {p['s_two_term']:.4f} mm/s; at infinity "
          f"{p['two_term'] * p['to_inf']:+.4f} +- {p['s_two_term'] * p['to_inf']:.4f}")
    for k, (lam, sig, frac) in p['single_couplings'].items():
        extra = f" -> Clipper {p['single'][k]:+.4f} mm/s (at infinity {p['single'][k] * p['to_inf']:+.4f})" if k in SINGLE else ' (not used)'
        print(f"  {k:10s} coupling ({lam / 1e3:.3f} +- {sig / 1e3:.3f}) e3, measured to {frac * 100:.1f}%{extra}")
    print(f"selection rule, Eq. (5): {p['rule']:.1f} (delta_in {g['din']:+.2f}, delta_out {g['dout']:+.2f})")
    print(f"Anderson relation: {p['anderson_inf']:+.4f} mm/s at infinity, {p['anderson_p']:+.4f} at perigee, {p['anderson_p'] / p['law4']:.1f} times Eq. (4)")
    near = astg.derived(fl['NEAR'])
    print(f"Eq. (6): ratio {ratio_eq6(near['rp'], near['vinf'], p['four']['lam']):.3f} at NEAR (leverage "
          f"{p['four']['leverage'][astg.ANOMALOUS.index('NEAR')] * 100:.0f}%), {ratio_eq6(p['rp'], p['vinf'], p['four']['lam']):.3f} at Clipper; "
          f"rp {p['rp'] / near['rp']:.3f} and Vinf {p['vinf'] / near['vinf']:.3f} times NEAR's")
    print('both-positive null encounters, law alone: ' + ', '.join(f"{k} {astg.astg_dvp(fl[k], p['four']['lam']):+.3f}" for k in ('Rosetta III', 'EPOXI II', 'Juno')))
    a = appendix_a()
    print('\nAppendix A')
    print(f"  r = ({a['r_vec'][0]:.3f}, {a['r_vec'][1]:.3f}, {a['r_vec'][2]:.3f}) km, v = ({a['v_vec'][0]:.6f}, {a['v_vec'][1]:.6f}, {a['v_vec'][2]:.6f}) km/s")
    print(f"  |r| {a['r']:.3f} km, |v| {a['v']:.5f} km/s, E {a['E']:.4f} km^2/s^2, Vinf {a['vinf']:.4f} km/s, eps {a['e']:.5f}, rp {a['rp']:.3f} km, "
          f"rp(1+eps) {a['semilatus']:.6e} m")
    print(f"  s_z {a['sz_in']:.6f} and {a['sz_out']:.6f}; dcos = cos {a['din']:.4f} - cos {a['dout']:.4f} = {a['cos_in']:.6f} - {a['cos_out']:.6f} = {a['dcos']:+.6f}")
    print(f"  Eq. (A3): {a['dvp']:+.4f} mm/s")
