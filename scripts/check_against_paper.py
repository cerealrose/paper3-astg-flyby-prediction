"""Recompute the numbers of the paper and compare each with the printed value at the printed precision.
Prints one line per check and a summary; exits with status 1 if any check fails.  Runs in about a minute and a half
(the covariance analyses dominate)."""
import sys
import numpy as np
import astg
from geometry import table1, frames
from predictions import predict, sensitivities, ratio_eq6, appendix_a, single_couplings
from covariance import table3, check_partials, coverage, arc_dependence
from perturbations import D_SUN, AU, S, A_SRP, shadow_times, shadow_effect, earth_radiation, drag, traj, R_EQ, propellant_fraction, budget
from thermal import thermal_cases
from outcomes import table5

results = []


def check(label, value, printed, decimals):
    ok = round(value, decimals) == round(printed, decimals) or abs(value - printed) <= 0.5 * 10**(-decimals) + 1e-12
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:62s} computed {value:+.{decimals + 3}f}  printed {printed:+.{decimals}f}")


def check_sci(label, value, printed, sig):
    ok = float(f"{value:.{sig - 1}e}") == printed
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:62s} computed {value:.{sig + 1}e}  printed {printed:.{sig - 1}e}")


def check_true(label, condition, detail=''):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label:62s} {detail}")


fl = astg.load_flybys()

# Section 1 and the couplings of the companion papers
check('kappa_A (1e-6)', astg.KAPPA_A * 1e6, 3.099, 3)
check('R_g (1e-3 m)', astg.RG * 1e3, 8.870, 3)
p = predict(fl)
check('lambda1, four encounters (1e3)', p['four']['lam'] / 1e3, 3.902, 3)
check('its standard error (1e3)', p['four']['se'] / 1e3, 0.576, 3)
check('lambda1, five encounters (1e3)', p['five']['lam'] / 1e3, 3.528, 3)
check('its standard error (1e3)', p['five']['se'] / 1e3, 0.913, 3)

# Section 2.1 and Table 1
t1 = table1(); g = t1['g']
check_true('Sec. 2.1 / Table 1 perigee 20:14:11.5 UTC', t1['utc'] == '20:14:11.5', t1['utc'])
check('Table 1 h_p (km)', g['hp'], 3234.4, 1)
check('Abstract h_p (km)', g['hp'], 3234, 0)
check('Table 1 r_p (Mm)', g['rp'] / 1e3, 9.6054, 4)
check('Table 1 V_inf (km/s)', g['vinf'], 11.559, 3)
check('Table 1 V_p (km/s)', g['vp'], 14.718, 3)
check('Table 1 eps', g['e'], 4.220, 3)
check('Table 1 inclination (deg)', g['incl'], 31.02, 2)
check('Table 1 delta_in (deg)', g['din'], 29.39, 2)
check('Table 1 delta_out (deg)', g['dout'], 30.53, 2)
check('Table 1 dcos', g['dcos'], 0.00991, 5)
check('Table 1 dsin', g['dsin'], 0.01719, 5)
check('Table 1 turn angle (deg)', g['turn'], 27.42, 2)
check('Table 1 perigee latitude (deg S)', -t1['lat'], 2.09, 2)
check('Table 1 perigee longitude (deg W)', -t1['lon'], 0.60, 2)
check('Table 1 shadow entry (min after perigee)', t1['shadow'][0], 3.7, 1)
check('Table 1 shadow exit (min after perigee)', t1['shadow'][1], 27.5, 1)
check('Table 1 Sun-Earth-probe angle (deg)', t1['sep'], 121.5, 1)
check('Sec. 2.1 geodetic height of the perigee point (km)', t1['geodetic_height'], 3227.3, 1)
check('Sec. 2.1 declination difference (deg)', g['dout'] - g['din'], 1.14, 2)
f = frames()
check('Sec. 2.1 pole of date from the ICRF pole (deg)', f['pole_offset'], 0.150, 3)
check_true('Sec. 2.1 below 0.08 deg at the companion encounters', max(f['companion_pole_offset'].values()) < 0.08,
           ', '.join(f"{k} {v:.4f}" for k, v in f['companion_pole_offset'].items()))
check('Sec. 2.1 ICRF delta_in (deg)', f['icrf']['din'], 29.389, 3)
check('Sec. 2.1 ICRF delta_out (deg)', f['icrf']['dout'], 30.606, 3)
check('Sec. 2.1 ICRF dcos larger by (%)', (f['dcos_ratio'] - 1) * 100, 7, 0)

# Section 2.2
s = sensitivities(p)
check_sci('Sec. 2.2 per km of perigee altitude (mm/s)', s['d_h'], -9.5e-6, 2)
check('Sec. 2.2 per degree of delta_in (mm/s)', s['d_in'], -0.044, 3)
check('Sec. 2.2 per degree of delta_out (mm/s)', s['d_out'], 0.045, 3)
check_true('Sec. 2.2 10 km or 0.01 deg moves the prediction by less than 1%', s['pct_10km'] < 1 and s['pct_001deg'] < 1,
           f"{s['pct_10km']:.2f}% and {s['pct_001deg']:.2f}%")
check('Sec. 2.2 Anderson per degree of delta_in (mm/s)', s['anderson_d_in'], -0.24, 2)
check('Sec. 2.2 Anderson per degree of delta_out (mm/s)', s['anderson_d_out'], 0.25, 2)

# Section 3 and Table 2
check('Eq. (4) ASTG law (mm/s)', p['law4'], 0.0504, 4)
check('Eq. (4) uncertainty (mm/s)', p['s_law4'], 0.0074, 4)
check('Abstract, law (mm/s)', p['law4'], 0.050, 3)
check('Abstract, its uncertainty (mm/s)', p['s_law4'], 0.007, 3)
check('Sec. 3.1 energy change Vp dVp (J/kg)', p['energy'], 0.741, 3)
check('Sec. 3.1 / Table 2 dVinf, four encounters (mm/s)', p['law4'] * p['to_inf'], 0.0641, 4)
check('Table 2 its uncertainty at infinity (mm/s)', p['s_law4'] * p['to_inf'], 0.0095, 4)
check('Sec. 3.1 / Table 2 five encounters (mm/s)', p['law5'], 0.0455, 4)
check('Table 2 its uncertainty (mm/s)', p['s_law5'], 0.0118, 4)
check('Table 2 five encounters at infinity (mm/s)', p['law5'] * p['to_inf'], 0.0580, 4)
check('Table 2 its uncertainty at infinity (mm/s)', p['s_law5'] * p['to_inf'], 0.0150, 4)
co, se = p['two_term_coef'], p['two_term_se']
check('Sec. 3.1 lambda_a (1e3)', co[0] / 1e3, 4.48, 2)
check('Sec. 3.1 sigma of lambda_a (1e3)', se[0] / 1e3, 1.53, 2)
check('Sec. 3.1 lambda_b (1e3)', co[1] / 1e3, 0.51, 2)
check('Sec. 3.1 sigma of lambda_b (1e3)', se[1] / 1e3, 1.21, 2)
check('Sec. 3.1 correlation of the two coefficients', p['two_term_corr'], 0.90, 2)
check('Sec. 3.1 / Table 2 two-term law (mm/s)', p['two_term'], 0.069, 3)
check('Table 2 its uncertainty (mm/s)', p['s_two_term'], 0.046, 3)
check('Table 2 two-term law at infinity (mm/s)', p['two_term'] * p['to_inf'], 0.088, 3)
check('Table 2 its uncertainty at infinity (mm/s)', p['s_two_term'] * p['to_inf'], 0.058, 3)
sc = single_couplings(fl)
check_true('Sec. 3.1 Galileo I, NEAR, Rosetta I measured to better than 8%', all(sc[k][2] < 0.08 for k in ('Galileo I', 'NEAR', 'Rosetta I')),
           ', '.join(f"{k} {sc[k][2] * 100:.1f}%" for k in ('Galileo I', 'NEAR', 'Rosetta I')))
for k, lam, sig, dec, sdec, clip in (('Galileo I', 7.19, 0.55, 2, 2, 0.093), ('NEAR', 3.967, 0.003, 3, 3, 0.051), ('Rosetta I', 1.58, 0.03, 2, 2, 0.020)):
    check(f'Sec. 3.1 {k} coupling (1e3)', sc[k][0] / 1e3, lam, dec)
    check(f'Sec. 3.1 {k} sigma (1e3)', sc[k][1] / 1e3, sig, sdec)
    check(f'Sec. 3.1 {k} at the Clipper geometry (mm/s)', p['single'][k], clip, 3)
check('Sec. 3.1 Cassini coupling (1e3)', sc['Cassini'][0] / 1e3, 47.9, 1)
check('Sec. 3.1 Cassini sigma (1e3)', sc['Cassini'][1] / 1e3, 24.0, 1)
check('Table 2 single-encounter range at infinity, low (mm/s)', p['single']['Rosetta I'] * p['to_inf'], 0.026, 3)
check('Table 2 single-encounter range at infinity, high (mm/s)', p['single']['Galileo I'] * p['to_inf'], 0.118, 3)
check_true('Sec. 3.2 selection rule predicts a null', p['rule'] == 0.0)
for k, v in (('Rosetta III', 0.247), ('EPOXI II', 0.273), ('Juno', 1.90)):
    check(f'Sec. 3.2 {k}, law alone (mm/s)', astg.astg_dvp(fl[k], p['four']['lam']), v, 3 if k != 'Juno' else 2)
check('Sec. 3.3 Anderson relation at infinity (mm/s)', p['anderson_inf'], 0.355, 3)
check('Sec. 3.3 Anderson relation at perigee (mm/s)', p['anderson_p'], 0.279, 3)
check('Sec. 3.3 Anderson / Eq. (4)', p['anderson_p'] / p['law4'], 5.5, 1)
near = astg.derived(fl['NEAR'])
check('Sec. 3.3 NEAR share of the leverage (%)', p['four']['leverage'][astg.ANOMALOUS.index('NEAR')] * 100, 90, 0)
check('Sec. 3.3 Eq. (6) at NEAR', ratio_eq6(near['rp'], near['vinf'], p['four']['lam']), 0.99, 2)
check('Sec. 3.3 Eq. (6) at Clipper', ratio_eq6(p['rp'], p['vinf'], p['four']['lam']), 0.18, 2)
check('Sec. 3.3 Clipper rp / NEAR rp', p['rp'] / near['rp'], 1.39, 2)
check('Sec. 3.3 Clipper Vinf / NEAR Vinf', p['vinf'] / near['vinf'], 1.69, 2)

# Section 4 and Table 3
check('Sec. 4 partials agree with finite differences to', check_partials() * 1e7, 3, 0)
c = coverage()
check('Sec. 4 no station in the last hours before perigee (h)', c['gap_before_h'], 4.2, 1)
check_true('Sec. 4 Madrid acquires 2 min after perigee', c['first_station'].startswith('Madrid') and round(c['first_after_min']) == 2,
           f"{c['first_station']} at {c['first_after_min']:.1f} min")
check_true('Sec. 4 Madrid keeps the spacecraft in view through the shadow', c['madrid_min_in_shadow'] > 10, f"lowest {c['madrid_min_in_shadow']:.1f} deg")
check('Sec. 4 Malargue elevation on the approach, up to (deg)', c['malargue_max'], 84, 0)
check_true('Sec. 4 radio path shorter than 3.1e6 km within three days', c['range_3d'] < 3.1e6, f"{c['range_3d']:.4e} km")
T3 = [(8025, 0.0053, 0.016), (8426, 0.0048, 0.015), (7967, 0.0054, 0.016), (7847, 0.0054, 0.016), (2563, 0.0109, 0.033),
      (13500, 0.0040, 0.012), (8025, 0.0026, 0.0079), (8025, 0.0105, 0.032), (8025, 0.0061, 0.018), (8634, 0.0050, 0.015),
      (7847, 0.0140, 0.042)]
rows3 = table3()
for r, (n, s1, s3) in zip(rows3, T3):
    check(f"Table 3 {r['case'][:36]} observations", r['n_obs'], n, 0)
    check(f"Table 3 {r['case'][:36]} 1 sigma (mm/s)", r['sigma'], s1, 4)
    check(f"Table 3 {r['case'][:36]} 3 sigma (mm/s)", r['three_sigma'], s3, 4 if s3 < 0.01 else 3)
check('Abstract, 1 sigma with pessimistic tracking (mm/s)', rows3[-1]['sigma'], 0.014, 3)
check('Sec. 4 baseline about a tenth of Eq. (4)', rows3[0]['sigma'] / p['law4'], 0.1, 1)
arcs = {d: s0 for d, s0, s1 in arc_dependence((1.0, 1.5, 2.0, 7.0))}
check_true('Sec. 4 law resolved at 3 sigma with +-1 day', 3 * arcs[1.0] < p['law4'], f"3 sigma {3 * arcs[1.0]:.4f} < {p['law4']:.4f}")
check_true('Sec. 4 lowest single-encounter value needs +-2 days', 3 * arcs[1.5] > p['single']['Rosetta I'] > 3 * arcs[2.0],
           f"3 sigma {3 * arcs[1.5]:.4f} (1.5 d), {3 * arcs[2.0]:.4f} (2 d) against {p['single']['Rosetta I']:.4f}")

# Section 5 and Table 4
tr = traj()
check('Sec. 5 propellant used by 13.35 m/s at 200 s, below 1% (%)', propellant_fraction() * 100, 0.68, 2)
dr = drag(tr=tr)
check_sci('Sec. 5.1 density, 130 sfu and Ap 15 (kg/m^3)', dr[1]['rho_perigee'], 2.4e-17, 2)
check_sci('Sec. 5.1 density, 250 sfu and Ap 100 (kg/m^3)', dr[2]['rho_perigee'], 6.1e-17, 2)
dmax = max(d['dv'] for d in dr)
check_sci('Sec. 5.1 drag impulse, at most (mm/s)', dmax, 1.7e-4, 2)
check('Sec. 5.1 factor below Eq. (4)', p['law4'] / dmax / 100, 3, 0)
check('Sec. 5.2 Sun distance (au)', D_SUN / AU, 0.9858, 4)
check('Sec. 5.2 solar flux (W/m^2)', S, 1400, 0)
check_sci('Sec. 5.2 solar-pressure acceleration (m/s^2)', A_SRP, 1.23e-7, 3)
st = shadow_times()
check('Sec. 5.2 penumbra entry (min)', st['pen_in'] / 60, 3.7, 1)
check('Sec. 5.2 altitude at entry (km)', st['alt_in'], 3665, 0)
check('Sec. 5.2 umbra entry (min)', st['umb_in'] / 60, 3.8, 1)
check('Sec. 5.2 umbra exit (min)', st['umb_out'] / 60, 27.1, 1)
check('Sec. 5.2 penumbra exit (min)', st['pen_out'] / 60, 27.5, 1)
check('Sec. 5.2 altitude at exit (km)', st['alt_out'], 17021, 0)
check('Sec. 5.2 eclipse (min)', (st['pen_out'] - st['pen_in']) / 60, 23.9, 1)
sh, nu, vdot = shadow_effect(tr=tr)
check('Eq. (7) shadow effect (mm/s)', sh, -0.122, 3)
check_true('Sec. 5.2 spacecraft recedes from the Sun in the shadow', np.mean(vdot[nu < 1]) > 0)
check_true('Sec. 5.2 shadow more than twice Eq. (4)', abs(sh) > 2 * p['law4'], f"{abs(sh):.4f} > {2 * p['law4']:.4f}")
check('Sec. 5.2 1% of the pressure model (mm/s)', 0.01 * abs(sh), 0.0012, 4)
check('Sec. 5.2 shadow radius 50 km larger (mm/s)', abs(shadow_effect(R_EQ + 50.0, tr=tr)[0] - sh), 0.0010, 4)
th = thermal_cases(); lags = [x['lag'] for x in th]
check('Sec. 5.3 thermal lag, smallest (mm/s)', max(lags), -0.0001, 4)
check('Sec. 5.3 thermal lag, largest (mm/s)', min(lags), -0.0012, 4)
check('Sec. 5.3 whole sunlit force kept through the shadow (mm/s)', max(x['full'] for x in th), 0.0072, 4)
er = earth_radiation(tr=tr)
check_sci('Sec. 5.4 albedo peak acceleration (m/s^2)', er['albedo']['peak'], 1.1e-9, 2)
check_sci('Sec. 5.4 infrared peak acceleration (m/s^2)', er['infrared']['peak'], 5.8e-9, 2)
check('Sec. 5.4 albedo (mm/s)', er['albedo']['dvp'], -0.0024, 4)
check('Sec. 5.4 infrared (mm/s)', er['infrared']['dvp'], 0.0015, 4)
check('Sec. 5.4 albedo after modelling, 20% (mm/s)', 0.2 * abs(er['albedo']['dvp']), 0.0005, 4)
check('Sec. 5.4 infrared after modelling, 20% (mm/s)', 0.2 * abs(er['infrared']['dvp']), 0.0003, 4)
b = budget(thermal_lag_max=max(abs(x) for x in lags))
check('Table 4 shadow after modelling (mm/s)', b['after']['shadow'], 0.0016, 4)
check('Table 4 atmospheric drag (mm/s)', b['drag'], -0.0002, 4)
check('Sec. 5.7 remaining effects in quadrature (mm/s)', b['total'], 0.0021, 4)
check('Abstract, remaining effects (mm/s)', b['total'], 0.002, 3)
check('Sec. 5.7 as a share of Eq. (4) (%)', b['total'] / p['law4'] * 100, 4, 0)
check_true('Sec. 5.7 below the baseline noise limit', b['total'] < rows3[0]['sigma'])

# Section 6 and Table 5
t5 = table5(p, b['total'])
bs, ps = t5['baseline'], t5['pessimistic']
check('Sec. 6 sigma_obs, baseline (mm/s)', bs['sigma_obs'], 0.00566, 5)
check('Sec. 6 sigma_obs, pessimistic (mm/s)', ps['sigma_obs'], 0.0142, 4)
for lab, key, vb, vp_ in (('null against the law, four encounters', 'null_law4', 5.4, 3.1), ('null against the law, five encounters', 'null_law5', 3.5, 2.5),
                          ('null against the lowest single value', 'null_low', 3.6, 1.4), ('law-sized change against the rule', 'law_vs_rule', 8.9, 3.5)):
    check(f'Table 5 {lab}, baseline (sigma)', bs[key], vb, 1)
    check(f'Table 5 {lab}, pessimistic (sigma)', ps[key], vp_, 1)
check('Table 5 null against Anderson, baseline (sigma)', bs['null_anderson'], 49, 0)
check('Table 5 null against Anderson, pessimistic (sigma)', ps['null_anderson'], 20, 0)
check('Table 5 law-sized change against Anderson, baseline (sigma)', bs['law_vs_anderson'], 40, 0)
check('Table 5 law-sized change against Anderson, pessimistic (sigma)', ps['law_vs_anderson'], 16, 0)
check_true('Sec. 6 Anderson more than 13 sigma from both ASTG predictions', min(bs['anderson_vs_top'], ps['anderson_vs_top']) > 13,
           f"smallest {min(bs['anderson_vs_top'], ps['anderson_vs_top']):.2f}")
check('Sec. 6 Anderson against a null reported to 0.1 mm/s (sigma)', t5['anderson_vs_null_at_0.1'], 2.8, 1)

# Appendix A
a = appendix_a()
for i, v in enumerate((2556.949, -9252.286, -348.478)):
    check(f'App. A r[{i}] (km)', a['r_vec'][i], v, 3)
for i, v in enumerate((12.238508, 3.097259, 7.565488)):
    check(f'App. A v[{i}] (km/s)', a['v_vec'][i], v, 6)
check('App. A |r| (km)', a['r'], 9605.426, 3)
check('App. A |v| (km/s)', a['v'], 14.71770, 5)
check('App. A specific energy (km^2/s^2)', a['E'], 66.8079, 4)
check('App. A V_inf (km/s)', a['vinf'], 11.5592, 4)
check('App. A eps', a['e'], 4.21986, 5)
check('App. A r_p (km)', a['rp'], 9605.426, 3)
check('App. A r_p (1 + eps) (1e7 m)', a['semilatus'] / 1e7, 5.013897, 6)
check('App. A s_z incoming', a['sz_in'], 0.490801, 6)
check('App. A s_z outgoing', a['sz_out'], 0.507995, 6)
check('App. A delta_in (deg)', a['din'], 29.3932, 4)
check('App. A delta_out (deg)', a['dout'], 30.5304, 4)
check('App. A cos delta_in', a['cos_in'], 0.871272, 6)
check('App. A cos delta_out', a['cos_out'], 0.861360, 6)
check('App. A dcos', a['dcos'], 0.009912, 6)
check('App. A Eq. (A3) (mm/s)', a['dvp'], 0.0504, 4)

n_fail = results.count(False)
print(f"\n{len(results) - n_fail} of {len(results)} checks pass")
sys.exit(1 if n_fail else 0)
