"""Shared geometry of the Europa Clipper Earth gravity assist of 3 December 2026.

States are read from data/horizons_states.csv: the geocentric state of Europa Clipper (NAIF -159) at closest
approach, 2026-12-03 20:15:20.7 TDB, in the ICRF and in the Earth's true equator and node of date, and the
geocentric position of the Sun at the same instant, all from the JPL Horizons trajectory segment
trj_260808-270123-dco2609230549-cruise019-pred-OD092 (orbit determination with tracking through
2026-09-22; Horizons record revised 2026-09-30).  horizons_clipper.py fetches them again.

Units inside this module are km, km/s and degrees unless a name says otherwise."""
import csv
import os
import numpy as np
import astg

GM = astg.GM / 1e9              # km^3 s^-2
RE = astg.R_EARTH / 1e3         # km, mean radius of Anderson et al. (2008)
OMEGA = astg.OMEGA
D2R = np.pi / 180
TDB_UTC = 69.184                # s, TT - TAI = 32.184 s plus 37 leap seconds
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')


def _states():
    out = {}
    with open(os.path.join(DATA, 'horizons_states.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            r = np.array([float(row[k]) for k in ('x_km', 'y_km', 'z_km')])
            v = np.array([float(row[k]) for k in ('vx_km_s', 'vy_km_s', 'vz_km_s')]) if row['vx_km_s'] else None
            out[row['label']] = (float(row['jd_tdb']), r, v)
    return out


_S = _states()
JD_CA_TDB, R_CA, V_CA = _S['clipper_perigee_icrf']
_, R_DATE, V_DATE = _S['clipper_perigee_equator_of_date']
_, SUN, _ = _S['sun_at_perigee_icrf']
JD_CA_UTC = JD_CA_TDB - TDB_UTC / 86400


def asymptotes(r, v):
    """Osculating hyperbola of a geocentric state: V_inf, eccentricity, perigee radius, inclination, the asymptotic
    velocity directions of Eq. (3) and their declinations (deg) in the frame of the state."""
    h = np.cross(r, v); evec = np.cross(v, h) / GM - r / np.linalg.norm(r)
    e = np.linalg.norm(evec); E = 0.5 * v @ v - GM / np.linalg.norm(r)
    p = h @ h / GM; rp = p / (1 + e)
    p_hat = evec / e; h_hat = h / np.linalg.norm(h); q_hat = np.cross(h_hat, p_hat)
    s_in = p_hat / e + np.sqrt(e * e - 1) / e * q_hat
    s_out = -p_hat / e + np.sqrt(e * e - 1) / e * q_hat
    dec = lambda u: np.degrees(np.arcsin(u[2]))
    return dict(vinf=np.sqrt(2 * E), E=E, e=e, rp=rp, p=p, incl=np.degrees(np.arccos(h_hat[2])), din=dec(s_in), dout=dec(s_out),
                p_hat=p_hat, q_hat=q_hat, h_hat=h_hat, s_in=s_in, s_out=s_out, turn=np.degrees(2 * np.arcsin(1 / e)))


def encounter(frame='date'):
    """Quantities of Table 1 from the state in the equator of date (frame='date') or the ICRF (frame='icrf')."""
    r, v = (R_DATE, V_DATE) if frame == 'date' else (R_CA, V_CA)
    g = asymptotes(r, v)
    g['hp'] = g['rp'] - RE
    g['vp'] = np.sqrt(g['vinf']**2 + 2 * GM / g['rp'])
    g['dcos'] = np.cos(g['din'] * D2R) - np.cos(g['dout'] * D2R)
    g['dsin'] = np.sin(g['dout'] * D2R) - np.sin(g['din'] * D2R)
    return g


def law(lam, hp_km, din, dout, vinf_kms):
    """Eq. (2) of the paper in mm/s for a perigee altitude, declinations and V_inf."""
    rp = (RE + hp_km) * 1e3; v = vinf_kms * 1e3
    e = 1 + rp * v**2 / astg.GM; vp = np.sqrt(v**2 + 2 * astg.GM / rp)
    return lam * astg.RG * (np.cos(din * D2R) - np.cos(dout * D2R)) / (2 * rp * (1 + e)) * vp * 1e3


def anderson_p(din, dout, vinf_kms, vp_kms):
    """Anderson relation converted to perigee, kappa_A dcos Vinf^2 / Vp, mm/s."""
    return astg.KAPPA_A * (np.cos(din * D2R) - np.cos(dout * D2R)) * vinf_kms**2 / vp_kms * 1e6


def gmst_deg(jd_ut1):
    jd0 = np.floor(jd_ut1 - 0.5) + 0.5; hh = (jd_ut1 - jd0) * 24; tu = (jd0 - 2451545.0) / 36525
    g0 = (24110.54841 + 8640184.812866 * tu + 0.093104 * tu**2 - 6.2e-6 * tu**3) % 86400 / 240
    return (g0 + 1.00273790935 * hh * 15) % 360


def precession_angles(jd_tt):
    """IAU 1976 precession angles zeta, z, theta (rad) from J2000 to date."""
    T = (jd_tt - 2451545.0) / 36525; as2r = np.pi / 180 / 3600
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * as2r
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * as2r
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * as2r
    return zeta, z, theta


def precession_matrix(jd_tt):
    def R2(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    def R3(a): c, s = np.cos(a), np.sin(a); return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    zeta, z, theta = precession_angles(jd_tt)
    return R3(-z) @ R2(theta) @ R3(-zeta)


def geodetic(r_icrf, jd_ut1, jd_tt=None, precess=True):
    """Geodetic latitude, east longitude in (-180, 180] and height (km) on WGS84 of an ICRF position, with
    precession to date (nutation, below 0.005 deg, and polar motion neglected)."""
    r = precession_matrix(jd_tt if jd_tt is not None else jd_ut1) @ r_icrf if precess else np.asarray(r_icrf)
    a, f = 6378.137, 1 / 298.257223563; e2 = f * (2 - f)
    th = gmst_deg(jd_ut1) * D2R
    x = np.cos(th) * r[0] + np.sin(th) * r[1]; y = -np.sin(th) * r[0] + np.cos(th) * r[1]; z = r[2]
    lon = np.degrees(np.arctan2(y, x)); pxy = np.hypot(x, y); lat = np.arctan2(z, pxy * (1 - e2))
    for _ in range(20):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2); hgt = pxy / np.cos(lat) - N
        lat = np.arctan2(z, pxy * (1 - e2 * N / (N + hgt)))
    return np.degrees(lat), (lon + 180) % 360 - 180, hgt


def pole_of_date():
    """Pole of the true equator of date in the ICRF, from the rotation that maps the ICRF state onto the
    equator-of-date state (orthogonal Procrustes on r, v and r x v)."""
    A = np.vstack([R_CA / np.linalg.norm(R_CA), V_CA / np.linalg.norm(V_CA), np.cross(R_CA, V_CA) / np.linalg.norm(np.cross(R_CA, V_CA))])
    B = np.vstack([R_DATE / np.linalg.norm(R_DATE), V_DATE / np.linalg.norm(V_DATE), np.cross(R_DATE, V_DATE) / np.linalg.norm(np.cross(R_DATE, V_DATE))])
    U, _, Vt = np.linalg.svd(B.T @ A); Rm = U @ Vt
    pole = Rm.T @ np.array([0.0, 0.0, 1.0])
    return pole, np.abs(B - (Rm @ A.T).T).max()


def utc_string(jd_utc, day0_jd=2461377.5):
    secs = (jd_utc - day0_jd) * 86400
    return '%02d:%02d:%04.1f' % (secs // 3600, secs % 3600 // 60, secs % 60)
