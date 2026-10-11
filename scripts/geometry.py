"""Geometry of the encounter (Section 2.1, Table 1, Eq. 3).

The osculating hyperbola at perigee follows from the Horizons state with the monopole GM only, as for the
encounters of the companion papers; altitudes are above the 6371 km mean radius.  Declinations are measured from
the Earth's true equator of date (Horizons REF_PLANE='BODY'); the ICRF values are printed for comparison, with the
angle between the two poles.  The perigee point is computed with precession to date (IAU 1976)."""
import numpy as np
import clipper as cl
from perturbations import shadow_times, sun_earth_probe_angle

COMPANION_DATES = {'Galileo I': 2448233.5, 'Juno': 2456574.5}     # first and last companion encounters (JD)
NUTATION_MAX = 10.0             # arcsec, bound on the nutation of the pole (9.2 arcsec principal term and smaller terms)


def table1():
    g = cl.encounter('date')
    lat, lon, hgt = cl.geodetic(cl.R_CA, cl.JD_CA_UTC, jd_tt=cl.JD_CA_TDB)
    s = shadow_times()
    return dict(g=g, utc=cl.utc_string(cl.JD_CA_UTC), lat=lat, lon=lon, geodetic_height=hgt,
                shadow=(s['pen_in'] / 60, s['pen_out'] / 60), sep=sun_earth_probe_angle())


def frames():
    """Declinations in the ICRF and in the equator of date, the pole offset at this encounter, and the
    angle between the ICRF pole and the pole of date at the first and last companion encounters, bounded by the
    precession angle theta (IAU 1976) plus the largest nutation of the pole."""
    gi, gd = cl.encounter('icrf'), cl.encounter('date')
    pole, resid = cl.pole_of_date()
    companion = {k: abs(np.degrees(cl.precession_angles(jd)[2])) + NUTATION_MAX / 3600 for k, jd in COMPANION_DATES.items()}
    return dict(icrf=gi, date=gd, pole_offset=np.degrees(np.arccos(pole[2])), rotation_residual=resid,
                dcos_ratio=gi['dcos'] / gd['dcos'], companion_pole_offset=companion)


if __name__ == '__main__':
    t = table1(); g = t['g']
    print('Table 1. Geometry of the Europa Clipper Earth gravity assist (equator of date)')
    print(f"  perigee {t['utc']} UTC (JD TDB {cl.JD_CA_TDB:.9f})")
    print(f"  hp {g['hp']:.1f} km   rp {g['rp'] / 1e3:.4f} Mm   Vinf {g['vinf']:.3f} km/s   Vp {g['vp']:.3f} km/s")
    print(f"  eps {g['e']:.3f}   inclination {g['incl']:.2f} deg   turn angle {g['turn']:.2f} deg")
    print(f"  delta_in {g['din']:+.2f}   delta_out {g['dout']:+.2f}   dcos {g['dcos']:+.5f}   dsin {g['dsin']:+.5f}   (difference {g['dout'] - g['din']:.2f} deg)")
    print(f"  perigee point {abs(t['lat']):.2f} {'S' if t['lat'] < 0 else 'N'}, {abs(t['lon']):.2f} {'W' if t['lon'] < 0 else 'E'}, geodetic height {t['geodetic_height']:.1f} km")
    print(f"  Earth shadow {t['shadow'][0]:+.1f} to {t['shadow'][1]:+.1f} min after perigee; Sun-Earth-probe angle {t['sep']:.1f} deg")
    f = frames()
    print(f"\nICRF: delta_in {f['icrf']['din']:+.3f}, delta_out {f['icrf']['dout']:+.3f}, dcos {f['icrf']['dcos']:+.6f}, "
          f"{(f['dcos_ratio'] - 1) * 100:.0f}% larger than in the equator of date ({f['date']['dcos']:+.6f})")
    print(f"pole of date {f['pole_offset']:.3f} deg from the ICRF pole (rotation residual {f['rotation_residual']:.0e}); "
          f"at the companion encounters at most " + ', '.join(f"{k} {v:.3f} deg" for k, v in f['companion_pole_offset'].items()))
