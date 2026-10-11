"""Detectability of a velocity change at the Europa Clipper perigee (Section 4, Table 3, Figure 2).

As for Juno in the companion paper: tracking before perigee fixes the trajectory and tracking after it measures an
impulsive change dV at perigee.
  * Estimated parameters: the perigee state (6), the three components of dV, and in two cases a solar-pressure
    scale factor; the solar-pressure partials include the Earth's shadow (perturbations.illumination).
  * Observable: two-way range rate every 60 s from whichever station sees the spacecraft highest above the mask,
    one 34 m antenna per Deep Space Network complex, with the ESA stations added in one case; white noise.
  * Partials from the two-body variational equations, checked against finite differences by check_partials().
  * Covariance P = (H^T W H)^-1; the printed number is the 1-sigma uncertainty of dV along the perigee velocity.
Run time is about half a minute for Table 3 and about as long again for the arc-length curve of Figure 2b."""
import numpy as np
from scipy.integrate import solve_ivp
import clipper as cl
from perturbations import illumination, S_HAT, A_SRP

GM, OMEGA = cl.GM, cl.OMEGA
A_WGS, F_WGS = 6378.137, 1 / 298.257223563
DSN = {'Goldstone DSS-25': (35.3376, 243.1246, 0.96), 'Canberra DSS-34': (-35.3985, 148.9819, 0.69),
       'Madrid DSS-54': (40.4256, 355.7459, 0.84)}
ESA = {'New Norcia (ESA)': (-31.0482, 116.1915, 0.25), 'Malargue (ESA)': (-35.7760, 290.6713, 1.55),
       'Cebreros (ESA)': (40.4527, 355.6326, 0.79)}
CLIPPER_SRP = A_SRP / 1e3          # km s^-2, nominal solar-pressure acceleration
CASES = [('Baseline', {}),
         ('Elevation mask 6 deg', {'mask_deg': 6.0}),
         ('2 h tracking gap at perigee', {'gap_h': 1.0}),
         ('6 h tracking gap at perigee', {'gap_h': 3.0}),
         ('Arc of +-1 day', {'days': 1.0}),
         ('Arc of +-5 days', {'days': 5.0}),
         ('Noise 0.05 mm/s', {'noise_mm': 0.05}),
         ('Noise 0.2 mm/s', {'noise_mm': 0.2}),
         ('Solar-pressure scale factor estimated', {'est_srp': True}),
         ('ESA stations at Malargue, New Norcia and Cebreros added', {'stations': {**DSN, **ESA}}),
         ('6 h gap, solar pressure estimated, noise 0.2 mm/s', {'gap_h': 3.0, 'est_srp': True, 'noise_mm': 0.2})]
ARCS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0)


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon); e2 = F_WGS * (2 - F_WGS); N = A_WGS / np.sqrt(1 - e2 * np.sin(la)**2)
    return (np.array([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - e2) + h) * np.sin(la)]),
            np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]))


def gmst(jd_ut1):
    return np.radians((280.46061837 + 360.98564736629 * (jd_ut1 - 2451545.0)) % 360)


def rotation(t):
    th = gmst(cl.JD_CA_UTC + t / 86400); c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def deriv(t, y, srp_hat):
    r = y[:3]; rn = np.linalg.norm(r)
    G = GM * (3 * np.outer(r, r) / rn**5 - np.eye(3) / rn**3)
    A = np.zeros((6, 6)); A[:3, 3:] = np.eye(3); A[3:, :3] = G
    Phi = y[6:42].reshape(6, 6); S = y[42:48]; dS = A @ S; dS[3:] += srp_hat * CLIPPER_SRP * illumination(r)
    return np.concatenate([y[3:6], -GM * r / rn**3, (A @ Phi).ravel(), dS])


_cache = {}


def propagate(days):
    """Forward and backward solutions with the state transition matrix and the solar-pressure sensitivity."""
    if days not in _cache:
        x0 = np.concatenate([cl.R_CA, cl.V_CA]); y0 = np.concatenate([x0, np.eye(6).ravel(), np.zeros(6)])
        _cache[days] = [solve_ivp(deriv, (0, s * days * 86400), y0, args=(-S_HAT,), rtol=1e-11, atol=1e-12,
                                  dense_output=True, method='DOP853', max_step=60.0) for s in (1, -1)]
    return _cache[days]


def design(days=3.0, count=60.0, gap_h=0.0, mask_deg=10.0, est_srp=False, stations=None, span=None):
    """Rows of the design matrix (unit noise) and, for each, the time, station and elevation."""
    stations = stations or DSN
    fw, bw = propagate(span or max(days, 5.0))
    sta = {k: ecef(*v) for k, v in stations.items()}
    npar = 10 if est_srp else 9
    rows, info = [], []
    for t in np.arange(-days * 86400, days * 86400 + 1, count):
        if abs(t) < gap_h * 3600 or t == 0:
            continue
        y = (fw if t > 0 else bw).sol(t); r, v = y[:3], y[3:6]; Phi = y[6:42].reshape(6, 6); S = y[42:48]
        R = rotation(t)
        best = None
        for k, (pe, up) in sta.items():
            rs = R @ pe; vs = np.cross([0, 0, OMEGA], rs); d = r - rs; rho = np.linalg.norm(d)
            el = np.degrees(np.arcsin(d @ (R @ up) / rho))
            if el >= mask_deg and (best is None or el > best[0]):
                best = (el, k, d, v - vs, rho)
        if best is None:
            continue
        el, k, d, dv, rho = best; rr = d @ dv / rho
        dh = np.concatenate([dv / rho - rr * d / rho**2, d / rho])
        h = np.zeros(npar); h[:6] = dh @ Phi
        if t > 0:
            h[6:9] = dh @ Phi[:, 3:6]
        if est_srp:
            h[9] = dh @ S
        rows.append(h); info.append((t, k, el))
    return np.array(rows), info


def analyse(noise_mm=0.1, **kw):
    """1-sigma uncertainty (mm/s) of dV along the perigee velocity, the number of observations and their record."""
    H, info = design(**kw)
    N = H.T @ H; sc = np.sqrt(np.diag(N)); sig = noise_mm * 1e-6
    P = np.linalg.inv(N / np.outer(sc, sc)) / np.outer(sc, sc) * sig**2
    u = cl.V_CA / np.linalg.norm(cl.V_CA)
    return np.sqrt(u @ P[6:9, 6:9] @ u) * 1e6, len(info), info


def table3(cases=CASES):
    out = []
    for name, kw in cases:
        s, n, _ = analyse(**kw)
        out.append(dict(case=name, n_obs=n, sigma=s, three_sigma=3 * s))
    return out


def arc_dependence(arcs=ARCS):
    """1-sigma uncertainty against the half-length of the arc, without and with the solar-pressure scale factor."""
    return [(d, analyse(days=d, span=max(arcs))[0], analyse(days=d, est_srp=True, span=max(arcs))[0]) for d in arcs]


def range_rate(x0, t, station, dv=np.zeros(3)):
    f = lambda tt, y: np.concatenate([y[3:], -GM * y[:3] / np.linalg.norm(y[:3])**3])
    y0 = x0.copy()
    if t > 0:
        y0[3:] += dv
    y = solve_ivp(f, (0, t), y0, rtol=1e-13, atol=1e-13, method='DOP853').y[:, -1]
    R = rotation(t); pe, up = ecef(*{**DSN, **ESA}[station])
    rs = R @ pe; vs = np.cross([0, 0, OMEGA], rs); d = y[:3] - rs; dvv = y[3:] - vs
    return d @ dvv / np.linalg.norm(d)


def check_partials(samples=8):
    """Largest relative difference between the variational partials and central finite differences, over a
    sample of observations of a +-1 day arc with hourly points."""
    H, info = design(days=1.0, count=3600.0)
    x0 = np.concatenate([cl.R_CA, cl.V_CA]); worst = 0.0
    for i in range(0, len(info), max(1, len(info) // samples)):
        t, k, el = info[i]
        for j, eps in enumerate([1e-3] * 3 + [1e-6] * 3):
            e = np.zeros(6); e[j] = eps
            fd = (range_rate(x0 + e, t, k) - range_rate(x0 - e, t, k)) / (2 * eps)
            worst = max(worst, abs(fd - H[i, j]) / max(abs(H[i, j]), 1e-12))
        if t > 0:
            for j in range(3):
                e = np.zeros(3); e[j] = 1e-6
                fd = (range_rate(x0, t, k, e) - range_rate(x0, t, k, -e)) / 2e-6
                worst = max(worst, abs(fd - H[i, 6 + j]) / max(abs(H[i, 6 + j]), 1e-12))
    return worst


def coverage():
    """Tracking geometry near perigee: the gap before perigee, the first station after it, the highest elevation
    at Malargue on the approach, Madrid's lowest elevation through the shadow, and the range at +-3 days."""
    _, _, info = analyse()
    t_obs = np.array([t for t, k, el in info])
    before = t_obs[t_obs < 0].max(); after = t_obs[t_obs > 0].min()
    first = [k for t, k, el in info if t == after][0]
    fw, bw = propagate(5.0)
    pe_m, up_m = ecef(*ESA['Malargue (ESA)']); pe_d, up_d = ecef(*DSN['Madrid DSS-54'])

    def elev(t, pe, up):
        r = (fw if t >= 0 else bw).sol(t)[:3]; R = rotation(t); d = r - R @ pe
        return np.degrees(np.arcsin(d @ (R @ up) / np.linalg.norm(d)))
    mal = max(elev(t, pe_m, up_m) for t in np.arange(-6 * 3600, 0, 60.0))
    from perturbations import shadow_times
    s = shadow_times()
    mad = min(elev(t, pe_d, up_d) for t in np.arange(s['pen_in'], s['pen_out'], 30.0))
    rng = max(np.linalg.norm(fw.sol(3 * 86400)[:3]), np.linalg.norm(bw.sol(-3 * 86400)[:3]))
    return dict(gap_before_h=-before / 3600, first_after_min=after / 60, first_station=first, malargue_max=mal,
                madrid_min_in_shadow=mad, range_3d=rng)


if __name__ == '__main__':
    print(f"nominal solar-pressure acceleration {CLIPPER_SRP * 1e3:.3e} m/s^2")
    print(f"partials against finite differences: largest relative difference {check_partials():.1e}")
    c = coverage()
    print(f"no station above the mask for the last {c['gap_before_h']:.2f} h before perigee; {c['first_station']} acquires "
          f"{c['first_after_min']:.0f} min after perigee and stays above {c['madrid_min_in_shadow']:.0f} deg through the shadow; "
          f"Malargue reaches {c['malargue_max']:.0f} deg on the approach; range at +-3 days {c['range_3d']:.3e} km")
    print('\nTable 3. Noise-only uncertainty of an along-track velocity change at perigee')
    for r in table3():
        print(f"  {r['case']:56s} {r['n_obs']:6d}  {r['sigma']:.4f}  {r['three_sigma']:.4f}")
    print('\nFigure 2b: arc half-length (days), 1 sigma, 1 sigma with the solar-pressure scale factor')
    for d, s0, s1 in arc_dependence():
        print(f"  {d:3.1f}  {s0:.4f}  {s1:.4f}")
