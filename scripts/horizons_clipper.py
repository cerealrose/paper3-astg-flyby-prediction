"""Fetch the Europa Clipper geocentric state at closest approach of the 3 December 2026 Earth gravity assist, and
the geocentric position of the Sun at that instant, from the JPL Horizons API (needs network access and the
requests package; runs in Google Colab).  The output has the columns of data/horizons_states.csv.

The perigee is found on a 1-minute grid, then as the zero of the range rate, interpolated linearly on a 2-minute
window sampled every second and rounded to 0.1 s; the states are then requested at that instant.  States are
geocentric, TDB, in the ICRF (REF_PLANE='FRAME') and in the frame of the Earth's true equator and node of date
(REF_PLANE='BODY').  Horizons serves its most recent trajectory solution, so a solution later than the one used in
the paper (trj_260808-270123-dco2609230549-cruise019-pred-OD092) gives slightly different numbers; the scripts can
be rerun on the new rows."""
import numpy as np
import requests

API = 'https://ssd.jpl.nasa.gov/api/horizons.api'


def query(command, extra, plane, table='3'):
    p = {'format': 'json', 'COMMAND': f"'{command}'", 'OBJ_DATA': "'NO'", 'MAKE_EPHEM': "'YES'", 'EPHEM_TYPE': "'VECTORS'",
         'CENTER': "'500@399'", 'REF_PLANE': f"'{plane}'", 'REF_SYSTEM': "'ICRF'", 'VEC_TABLE': f"'{table}'", 'OUT_UNITS': "'KM-S'",
         'CSV_FORMAT': "'YES'", 'VEC_LABELS': "'NO'", 'TIME_DIGITS': "'FRACSEC'"}
    p.update(extra)
    res = requests.get(API, params=p, timeout=120).json()['result']
    rows = [l.split(',') for l in res.split('$$SOE')[1].split('$$EOE')[0].strip().splitlines()]
    jd = np.array([float(r[0]) for r in rows]); date = [r[1].strip() for r in rows]
    s = np.array([[float(x) for x in r[2:8]] for r in rows])
    rr = np.array([float(r[10]) for r in rows]) if table == '3' else None
    return jd, date, s, rr


if __name__ == '__main__':
    span = lambda a, b, step: {'START_TIME': f"'{a}'", 'STOP_TIME': f"'{b}'", 'STEP_SIZE': f"'{step}'"}
    jd, date, s, rr = query('-159', span('2026-12-03 19:00', '2026-12-03 21:00', '1m'), 'FRAME')
    i = int(np.argmin(np.linalg.norm(s[:, :3], axis=1)))
    jd, date, s, rr = query('-159', span(date[i - 1].replace('A.D. ', ''), date[i + 1].replace('A.D. ', ''), '120'), 'FRAME')
    k = int(np.nonzero((rr[:-1] < 0) & (rr[1:] >= 0))[0][0])
    jd_p = jd[k] - rr[k] * (jd[k + 1] - jd[k]) / (rr[k + 1] - rr[k])
    jd_p = np.floor(jd_p) + np.round((jd_p - np.floor(jd_p)) * 864000) / 864000      # nearest 0.1 s
    tlist = {'TLIST_TYPE': "'JD'", 'TLIST': f"'{jd_p:.9f}'"}
    print('label,frame,jd_tdb,epoch_tdb,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s')
    for label, plane, frame in (('clipper_perigee_icrf', 'FRAME', 'ICRF (REF_PLANE=FRAME)'),
                                ('clipper_perigee_equator_of_date', 'BODY', 'Earth true equator and node of date (REF_PLANE=BODY)')):
        _, d, sj, _ = query('-159', tlist, plane)
        print(f"{label},{frame},{jd_p:.9f},{d[0].replace('A.D. ', '')}," + ','.join(f'{x:.15E}' for x in sj[0]))
    _, d, sun, _ = query('10', tlist, 'FRAME', table='2')
    print(f"sun_at_perigee_icrf,ICRF (REF_PLANE=FRAME),{jd_p:.9f},{d[0].replace('A.D. ', '')}," + ','.join(f'{x:.15E}' for x in sun[0, :3]) + ',,,')
