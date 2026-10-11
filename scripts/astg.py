"""Shared constants, data loading and formulae for the ASTG Earth-flyby papers
(Nesvinga, Jones and Nyambuya 2026).

Quantities are SI unless a name says otherwise.  The flyby law is Eq. (12) of the
first paper,

    dVp/Vp = lambda1 * Rg * dcos / (2 * rp * (1 + eps)),   Rg = 2 GM / c0^2,

and the empirical relation of Anderson et al. (2008) is dVinf/Vinf = kappa_A * dcos,
with dcos = cos(delta_in) - cos(delta_out).
"""
import csv
import os
import numpy as np
from scipy import stats

GM = 3.986004418e14            # m^3 s^-2, IERS Conventions (2010)
R_EARTH = 6.371e6              # m, mean radius used by Anderson et al. (2008)
C0 = 299792458.0               # m s^-1
OMEGA = 7.292115e-5            # rad s^-1, sidereal rotation rate
RG = 2 * GM / C0**2            # m, 8.870e-3
KAPPA_A = 2 * OMEGA * R_EARTH / C0   # Anderson constant, 3.099e-6

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data', 'earth_flybys.csv')
ANOMALOUS = ['Galileo I', 'NEAR', 'Cassini', 'Rosetta I']
NULLS = ['MESSENGER', 'Rosetta II', 'Rosetta III', 'Juno', 'EPOXI I', 'EPOXI II', 'EPOXI III']


def load_flybys(path=DATA_FILE):
    """Rows of data/earth_flybys.csv as a dict keyed by spacecraft, numbers as floats (None if blank)."""
    out = {}
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for k, v in row.items():
                if k in ('spacecraft', 'date', 'class', 'anomaly_source', 'geometry_source'):
                    rec[k] = v
                else:
                    rec[k] = float(v) if v.strip() else None
            out[row['spacecraft']] = rec
    return out


def derived(rec):
    """Eq. (9) of the first paper: perigee radius, perigee speed, eccentricity, dcos and the regressor X."""
    rp = R_EARTH + rec['hp_km'] * 1e3
    vinf = rec['Vinf_km_s'] * 1e3
    vp = np.sqrt(vinf**2 + 2 * GM / rp)
    eps = 1 + rp * vinf**2 / GM
    dcos = np.cos(np.radians(rec['delta_in_deg'])) - np.cos(np.radians(rec['delta_out_deg']))
    dsin = np.sin(np.radians(rec['delta_out_deg'])) - np.sin(np.radians(rec['delta_in_deg']))
    semilatus = rp * (1 + eps)
    x = RG * dcos / (2 * semilatus)
    return dict(rp=rp, vinf=vinf, vp=vp, eps=eps, dcos=dcos, dsin=dsin, semilatus=semilatus, X=x)


def astg_dvp(rec, lam):
    """Perigee anomaly of the ASTG law, Eq. (12), in mm/s."""
    d = derived(rec)
    return lam * d['X'] * d['vp'] * 1e3


def anderson_dvinf(rec):
    """Anderson et al. (2008) relation at infinity, mm/s."""
    d = derived(rec)
    return KAPPA_A * d['dcos'] * d['vinf'] * 1e3


def anderson_dvp(rec):
    """Anderson et al. (2008) relation converted to perigee, dVp = kappa_A dcos Vinf^2 / Vp, mm/s."""
    d = derived(rec)
    return KAPPA_A * d['dcos'] * d['vinf']**2 / d['vp'] * 1e3


def fit_through_origin(x, y):
    """Ordinary least squares through the origin, y = lam x, with the statistics quoted in the papers."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); dof = n - 1
    lam = x @ y / (x @ x)
    res = y - lam * x
    s2 = res @ res / dof
    se = np.sqrt(s2 / (x @ x))
    t = lam / se
    p = 2 * stats.t.sf(abs(t), dof)
    r2_mean = 1 - res @ res / np.sum((y - y.mean())**2)
    r2_origin = 1 - res @ res / (y @ y)
    lev = x**2 / (x @ x)
    return dict(lam=lam, se=se, t=t, p=p, r2_mean=r2_mean, r2_origin=r2_origin, res=res, s2=s2, dof=dof, leverage=lev)


def anomalous_fit(flybys, dvp_override=None, names=ANOMALOUS):
    """Fit of the coupling to the named encounters; dvp_override replaces observed dVp (mm/s) by name."""
    dvp_override = dvp_override or {}
    x, y = [], []
    for k in names:
        d = derived(flybys[k])
        dvp = dvp_override.get(k, flybys[k]['dVp_mm_s'])
        x.append(d['X']); y.append(dvp * 1e-3 / d['vp'])
    return fit_through_origin(x, y)
