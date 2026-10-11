"""Reading the outcome (Section 6, Table 5).

sigma_obs combines the noise-only limits of Table 3 (baseline and the most pessimistic case) with what remains of
the conventional effects after modelling (Table 4, in quadrature).  Deviations are in units of the combined
uncertainty of the measurement and, for the law, the coupling, as in the companion papers."""
import numpy as np
from covariance import analyse
from perturbations import budget
from predictions import predict

PESSIMISTIC = dict(gap_h=3.0, est_srp=True, noise_mm=0.2)


def table5(p=None, floor=None):
    p = p or predict()
    floor = floor if floor is not None else budget()['total']
    low, mid, high = (p['single'][k] for k in ('Rosetta I', 'NEAR', 'Galileo I'))
    A = p['anderson_p']
    out = {}
    for name, kw in (('baseline', {}), ('pessimistic', PESSIMISTIC)):
        noise = analyse(**kw)[0]; so = np.hypot(noise, floor)
        out[name] = dict(noise=noise, sigma_obs=so,
                         null_law4=p['law4'] / np.hypot(so, p['s_law4']), null_law5=p['law5'] / np.hypot(so, p['s_law5']),
                         null_low=low / so, null_anderson=A / so, law_vs_rule=p['law4'] / so, law_vs_anderson=(A - p['law4']) / so,
                         anderson_vs_top=(A - high) / so)
    out['anderson_vs_null_at_0.1'] = A / 0.1
    out['floor'] = floor
    return out


if __name__ == '__main__':
    t = table5()
    print(f"conventional effects after modelling, in quadrature: {t['floor']:.4f} mm/s")
    for name in ('baseline', 'pessimistic'):
        r = t[name]
        print(f"\n{name}: noise {r['noise']:.5f}, sigma_obs {r['sigma_obs']:.5f} mm/s")
        print(f"  null observed:      law (four encounters) {r['null_law4']:.1f} sigma, law (five) {r['null_law5']:.1f}, "
              f"lowest single-encounter value {r['null_low']:.1f}, Anderson {r['null_anderson']:.1f}")
        print(f"  law-sized change:   selection rule {r['law_vs_rule']:.1f} sigma, Anderson {r['law_vs_anderson']:.1f}")
        print(f"  Anderson against the top of the single-encounter range: {r['anderson_vs_top']:.1f} sigma")
    print(f"\nreported to 0.1 mm/s, the Anderson relation is {t['anderson_vs_null_at_0.1']:.1f} sigma from a null")
