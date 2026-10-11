"""Conventional accelerations at the Europa Clipper encounter (Section 5, Table 4), each expressed as the change of
perigee speed dVp = dE / Vp that its energy change produces.

  5.1  atmospheric drag: NRLMSISE-00 (pymsis, version=0) on the vertical through the perigee point, integrated
       along the trajectory at the speed relative to the co-rotating atmosphere, 127 m^2, C_D = 2.2
  5.2  solar radiation pressure and the Earth's shadow: conical shadow, Earth radius 6378.137 km, solar radius
       696,000 km, Sun position from Horizons (data/horizons_states.csv); entry and exit by root finding
  5.4  Earth albedo and infrared pressure: Lambertian sphere, Bond albedo 0.293, emitted flux 239.6 W/m^2
       (Loeb et al. 2009), radial force on the arrays (Sun-pointed) and a 25 m^2 bus
  5.7  budget: what remains after modelling, in quadrature (thermal recoil from thermal.py)
Spacecraft (JPL press kit): arrays 102 m^2, launch mass 5800 kg; bus and high-gain antenna 25 m^2 projected and
a radiation-pressure coefficient of 1.2 are estimates.  The trajectory is a two-body propagation of the Horizons
perigee state over +-6 h."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import clipper as cl

C = 299792458.0                     # m/s
AU = 1.495978707e8                  # km
S1AU = 1360.8                       # W m^-2, total solar irradiance (Kopp and Lean 2011)
R_SUN, R_EQ = 696000.0, 6378.137    # km
M_SC, A_ARR, A_BUS, CR, CD = 5800.0, 102.0, 25.0, 1.2, 2.2
ALBEDO, EMITTED = 0.293, 239.6      # Loeb et al. (2009)
D_SUN = np.linalg.norm(cl.SUN); S_HAT = cl.SUN / D_SUN
S = S1AU * (AU / D_SUN)**2          # W m^-2 at the encounter
A_SRP = S / C * (A_ARR + A_BUS) * CR / M_SC      # m s^-2
VP = np.linalg.norm(cl.V_CA) * 1e3  # m/s
SPACE_WEATHER = ((70, 70, 4), (130, 130, 15), (250, 200, 100))     # F10.7, 81-day mean, Ap
TRAJECTORY_CORRECTIONS = 13.35      # m/s, launch to the clean-up after the Mars flyby (Cangahuala et al. 2026)
PROPELLANT = 2750.0                 # kg


def _f(t, y):
    return np.concatenate([y[3:], -cl.GM * y[:3] / np.linalg.norm(y[:3])**3])


def traj(t_span=6 * 3600, n=43201):
    """Two-body trajectory through perigee on a uniform grid: times (s), positions (km), velocities (km/s)."""
    ts = np.linspace(-t_span, t_span, n)
    y0 = np.concatenate([cl.R_CA, cl.V_CA])
    fw = solve_ivp(_f, (0, t_span), y0, t_eval=ts[ts >= 0], rtol=1e-12, atol=1e-9, method='DOP853')
    bw = solve_ivp(_f, (0, -t_span), y0, t_eval=ts[ts <= 0][::-1], rtol=1e-12, atol=1e-9, method='DOP853')
    t = np.concatenate([bw.t[::-1], fw.t[1:]]); y = np.concatenate([bw.y[:, ::-1], fw.y[:, 1:]], axis=1)
    return t, y[:3].T, y[3:].T


def illumination(r, r_eq=R_EQ):
    """Fraction of the solar disc visible from r (km): conical shadow of a spherical Earth."""
    d_s = cl.SUN - r; ds = np.linalg.norm(d_s); de = np.linalg.norm(r)
    a = np.arcsin(R_SUN / ds); b = np.arcsin(r_eq / de)
    c = np.arccos(np.clip(np.dot(d_s, -r) / (ds * de), -1, 1))
    if c >= a + b:
        return 1.0
    if c <= b - a:
        return 0.0
    if c <= a - b:
        return 1.0 - (b / a)**2
    x = (c * c + a * a - b * b) / (2 * c); y = np.sqrt(max(a * a - x * x, 0))
    area = a * a * np.arccos(x / a) + b * b * np.arccos((c - x) / b) - c * y
    return 1.0 - area / (np.pi * a * a)


def shadow_times(r_eq=R_EQ):
    """Penumbra entry, umbra entry and exit, penumbra exit (s after perigee) and the altitudes at the penumbra
    boundaries (km above the mean radius)."""
    sol = solve_ivp(_f, (0, 3600.0), np.concatenate([cl.R_CA, cl.V_CA]), rtol=1e-12, atol=1e-9, method='DOP853', dense_output=True)

    def angles(t):
        r = sol.sol(t)[:3]; d_s = cl.SUN - r; ds = np.linalg.norm(d_s); de = np.linalg.norm(r)
        return np.arcsin(R_SUN / ds), np.arcsin(r_eq / de), np.arccos(np.dot(d_s, -r) / (ds * de))
    pen = lambda t: (lambda a, b, c: c - (a + b))(*angles(t))
    umb = lambda t: (lambda a, b, c: c - (b - a))(*angles(t))
    t1, t2 = brentq(pen, 60, 900), brentq(umb, 60, 900)
    t3, t4 = brentq(umb, 900, 3000), brentq(pen, 900, 3000)
    alt = lambda t: np.linalg.norm(sol.sol(t)[:3]) - cl.RE
    return dict(pen_in=t1, umb_in=t2, umb_out=t3, pen_out=t4, alt_in=alt(t1), alt_out=alt(t4))


def shadow_effect(r_eq=R_EQ, tr=None):
    """Eq. (7): change of perigee speed (mm/s) from the solar pressure missing in the shadow."""
    t, R, V = tr or traj()
    nu = np.array([illumination(r, r_eq) for r in R])
    vdot = (V * 1e3) @ (-S_HAT)                                   # (-s).v, m/s
    return -np.sum((1 - nu) * A_SRP * vdot) * (t[1] - t[0]) / VP * 1e3, nu, vdot


def earth_radiation(tr=None):
    """Earth albedo and infrared pressure: peak acceleration (m/s^2) and change of perigee speed (mm/s)."""
    t, R, V = tr or traj()
    r_n = np.linalg.norm(R, axis=1); rhat = R / r_n[:, None]
    beta = np.arccos(np.clip(rhat @ S_HAT, -1, 1))                # Sun-Earth-spacecraft angle
    phase = (2 / (3 * np.pi)) * (np.sin(beta) + (np.pi - beta) * np.cos(beta))
    area = A_ARR * np.abs(rhat @ S_HAT) + A_BUS
    vr = np.einsum('ij,ij->i', rhat, V * 1e3)
    out = {}
    for name, F in (('albedo', ALBEDO * S * (cl.RE / r_n)**2 * phase), ('infrared', EMITTED * (cl.RE / r_n)**2)):
        a = F / C * area / M_SC
        out[name] = dict(peak=a.max(), dvp=np.sum(a * vr) * (t[1] - t[0]) / VP * 1e3)
    return out


def drag(tr=None):
    """Density at the perigee point and the drag impulse (mm/s) for the space-weather cases of SPACE_WEATHER."""
    import warnings
    import pymsis
    warnings.filterwarnings('ignore', module='pymsis')
    t, R, V = tr or traj()
    lat, lon, h = cl.geodetic(cl.R_CA, cl.JD_CA_UTC, precess=False)
    hs = np.arange(h, h + 4000.0, 10.0)
    r_n = np.linalg.norm(R, axis=1)
    vrel = V * 1e3 - np.cross([0, 0, cl.OMEGA], R * 1e3); vr2 = np.einsum('ij,ij->i', vrel, vrel)
    out = []
    for f107, f107a, ap in SPACE_WEATHER:
        rho = np.squeeze(pymsis.calculate(np.datetime64('2026-12-03T20:14:12'), lon, lat, hs, f107s=f107, f107as=f107a,
                                          aps=[[ap] * 7], version=0))[:, 0]
        rho_t = np.exp(np.interp(r_n - cl.RE, hs, np.log(rho), left=np.log(rho[0]), right=-np.inf))
        dv = 0.5 * CD * (A_ARR + A_BUS) / M_SC * np.sum(rho_t * vr2) * (t[1] - t[0])
        out.append(dict(f107=f107, ap=ap, rho_perigee=rho[0], dv=dv * 1e3))
    return out


def sun_earth_probe_angle():
    return np.degrees(np.arccos(cl.R_CA @ S_HAT / np.linalg.norm(cl.R_CA)))


def propellant_fraction(isp=200.0):
    """Fraction of the launch mass used by the trajectory corrections at a given specific impulse (s)."""
    return 1 - np.exp(-TRAJECTORY_CORRECTIONS / (isp * 9.80665))


def budget(thermal_lag_max=None):
    """Table 4: each effect unmodelled and after modelling (mm/s), and the quadrature sum after modelling."""
    tr = traj()
    sh, nu, vdot = shadow_effect(tr=tr); sh50 = shadow_effect(R_EQ + 50.0, tr=tr)[0]
    er = earth_radiation(tr=tr); dr = max(d['dv'] for d in drag(tr=tr))
    if thermal_lag_max is None:
        from thermal import thermal_cases
        thermal_lag_max = max(abs(c['lag']) for c in thermal_cases())
    after = dict(shadow=np.hypot(0.01 * sh, sh50 - sh), thermal=thermal_lag_max, albedo=0.2 * abs(er['albedo']['dvp']),
                 infrared=0.2 * abs(er['infrared']['dvp']), drag=dr)
    total = np.sqrt(sum(v**2 for v in after.values()))
    return dict(shadow=sh, shadow_50km=sh50 - sh, albedo=er['albedo'], infrared=er['infrared'], drag=-dr, after=after, total=total,
                receding=np.mean(vdot[nu < 1] / np.linalg.norm(tr[2][nu < 1] * 1e3, axis=1)))


if __name__ == '__main__':
    tr = traj()
    print(f"5.2  Sun distance {D_SUN / AU:.4f} au, solar flux {S:.1f} W/m^2, solar-pressure acceleration {A_SRP:.3e} m/s^2")
    s = shadow_times()
    print(f"     penumbra entry {s['pen_in'] / 60:.2f} min after perigee at {s['alt_in']:.0f} km; umbra {s['umb_in'] / 60:.2f} to "
          f"{s['umb_out'] / 60:.2f} min; penumbra exit {s['pen_out'] / 60:.2f} min at {s['alt_out']:.0f} km; eclipse {(s['pen_out'] - s['pen_in']) / 60:.2f} min")
    sh, nu, vdot = shadow_effect(tr=tr)
    sh50 = shadow_effect(R_EQ + 50.0, tr=tr)[0]
    print(f"     Eq. (7) shadow effect {sh:+.4f} mm/s; 1% of it {abs(sh) * 0.01:.4f}; shadow radius 50 km larger changes it by {sh50 - sh:+.4f}")
    print(f"     mean (-s).v_hat in the shadow {np.mean(vdot[nu < 1] / np.linalg.norm(tr[2][nu < 1] * 1e3, axis=1)):+.3f} (receding from the Sun)")
    print(f"     Sun-Earth-probe angle at perigee {sun_earth_probe_angle():.2f} deg")
    er = earth_radiation(tr=tr)
    for k, v in er.items():
        print(f"5.4  {k:9s} peak {v['peak']:.2e} m/s^2, perigee-speed change {v['dvp']:+.5f} mm/s, 20% of it {0.2 * abs(v['dvp']):.5f}")
    for d in drag(tr=tr):
        print(f"5.1  F10.7 {d['f107']:3d} sfu, Ap {d['ap']:3d}: density at perigee {d['rho_perigee']:.2e} kg/m^3, impulse {d['dv']:.2e} mm/s")
    print(f"5    trajectory corrections of {TRAJECTORY_CORRECTIONS} m/s use {propellant_fraction() * 100:.2f}% of the mass at Isp = 200 s")
    b = budget()
    print('\nTable 4. Conventional effects (mm/s): unmodelled / after modelling')
    print(f"  Earth shadow       {b['shadow']:+.4f} / {b['after']['shadow']:.4f}")
    print(f"  array thermal      (thermal.py)  / {b['after']['thermal']:.4f} at most")
    print(f"  Earth albedo       {b['albedo']['dvp']:+.4f} / {b['after']['albedo']:.4f}")
    print(f"  Earth infrared     {b['infrared']['dvp']:+.4f} / {b['after']['infrared']:.4f}")
    print(f"  atmospheric drag   {b['drag']:+.4f} / {b['drag']:+.4f}")
    print(f"  after modelling, in quadrature: {b['total']:.4f} mm/s")
