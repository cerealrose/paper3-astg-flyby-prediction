"""Thermal recoil of the solar arrays through the Earth's shadow (Section 5.3).

Two-node panel: front (cells) and back faces each carry half of an areal heat capacity c_A and are joined by a
conductance G chosen so that the front-to-back temperature difference in sunlight is dT; solar absorptance 0.90,
front emissivity 0.85, back emissivity eps_b.  The recoil force along the anti-Sun direction is
    F = (2/3)(A/c0) sigma (eps_f Tf^4 - eps_b Tb^4).
A navigation force model scales the sunlit radiation force by the illumination fraction nu(t); the part it misses,
F(t) - F_sun nu(t), is the lag of the panel temperatures at shadow entry and exit."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from perturbations import traj, illumination, S, S_HAT, A_ARR, M_SC, C

SIG = 5.670374419e-8
CA_VALUES = (1500.0, 2500.0, 4000.0)       # J m^-2 K^-1
DT_VALUES = (5.0, 20.0)                    # K
EPS_B_VALUES = (0.80, 0.85, 0.90)


def setup():
    t, R, V = traj(t_span=3 * 3600, n=21601)
    nu = np.array([illumination(r) for r in R])
    vp = np.linalg.norm(V[len(t) // 2]) * 1e3
    return t, nu, (V * 1e3) @ (-S_HAT), vp


def run(cA=2500.0, dT_ss=10.0, eps_f=0.85, eps_b=0.85, alpha=0.90, data=None):
    t, nu, vdot, vp = data or setup()
    c = cA / 2

    def steady(G):
        def f(Tb):
            Tf = Tb + eps_b * SIG * Tb**4 / G
            return alpha * S - eps_f * SIG * Tf**4 - eps_b * SIG * Tb**4
        Tb = brentq(f, 100, 500)
        return Tb + eps_b * SIG * Tb**4 / G, Tb
    G = brentq(lambda G: (lambda x: x[0] - x[1])(steady(G)) - dT_ss, 1.0, 1e4)
    Tf0, Tb0 = steady(G)

    def rhs(tt, y):
        Tf, Tb = y; n = np.interp(tt, t, nu)
        return [(alpha * S * n - eps_f * SIG * Tf**4 - G * (Tf - Tb)) / c, (G * (Tf - Tb) - eps_b * SIG * Tb**4) / c]
    sol = solve_ivp(rhs, (t[0], t[-1]), [Tf0, Tb0], t_eval=t, rtol=1e-9, atol=1e-9, max_step=5.0)
    Tf, Tb = sol.y
    F = (2 / 3) * A_ARR / C * SIG * (eps_f * Tf**4 - eps_b * Tb**4)
    Fss = F[0]; dt = t[1] - t[0]
    return dict(cA=cA, dT=dT_ss, eps_b=eps_b, G=G, Tf0=Tf0, Tb0=Tb0, Tmin=Tf.min(), F_sun=Fss,
                lag=np.sum((F - Fss * nu) * vdot) * dt / M_SC / vp * 1e3,
                full=Fss * np.sum((1 - nu) * vdot) * dt / M_SC / vp * 1e3)


def thermal_cases():
    data = setup()
    return [run(cA, dT, 0.85, eb, data=data) for cA in CA_VALUES for dT in DT_VALUES for eb in EPS_B_VALUES]


if __name__ == '__main__':
    cases = thermal_cases()
    for r in cases:
        print(f"c_A {r['cA']:5.0f}  dT {r['dT']:4.1f}  eps_b {r['eps_b']:.2f}:  G {r['G']:6.1f}  Tf {r['Tf0']:.1f}  Tb {r['Tb0']:.1f}  "
              f"Tmin {r['Tmin']:.0f} K  F_sun {r['F_sun'] * 1e6:5.1f} uN  lag {r['lag']:+.2e} mm/s  (force kept through the shadow {r['full']:+.4f})")
    lags = [r['lag'] for r in cases]
    print(f"\nlag of the panel temperatures: {max(lags):+.4f} to {min(lags):+.4f} mm/s; "
          f"whole sunlit force kept through the shadow at most {max(r['full'] for r in cases):+.4f} mm/s")
