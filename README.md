# Application of the ASTG-Model to Earth Flyby Encounters: On a Pre-Flyby Predictive Test of the ASTG

P. A. Nesvinga<sup>1,2</sup>, B. C. Jones<sup>3</sup> and G. G. Nyambuya<sup>1,2</sup>

<sup>1</sup> Fundamental Theoretical Physics Group, National University of Science and Technology, Bulawayo, Zimbabwe<br>
<sup>2</sup> Department of Applied Physics, National University of Science and Technology, Bulawayo, Zimbabwe<br>
<sup>3</sup> Department of Applied Mathematics, National University of Science and Technology, Bulawayo, Zimbabwe

Data and code for the third and final paper of the series, submitted to *Advances in Space Research* (2026). The paper
applies the ASTG flyby law and the hemispheric selection rule of the companion papers, ahead of the event, to the Earth
gravity assist of Europa Clipper on 3 December 2026. With both asymptotic declinations positive (+29.39° and +30.53° from
the Earth's equator of date) the rule predicts no anomaly; the law alone predicts +0.0504 ± 0.0074 mm/s at perigee and the
Anderson relation +0.279 mm/s. Deep Space Network Doppler would measure the change to 0.0053 mm/s (1σ).

## What the paper says is here

Appendix B of the paper reads:

> The Horizons state vectors, the osculating-element and prediction code of Sections 2 and 3, the covariance analysis of Section 4, the shadow, radiation-pressure, thermal and drag calculations of Section 5 and the code producing Figures 1–3 are available at https://github.com/cerealrose.

| Item in the paper | Where it is |
|---|---|
| Horizons state vectors | `data/horizons_states.csv`; `scripts/horizons_clipper.py` fetches them again |
| Osculating elements, Section 2 and Table 1 | `scripts/clipper.py`, `scripts/geometry.py` |
| Predictions, Sections 2.2 and 3, Table 2 and Appendix A | `scripts/predictions.py` |
| Covariance analysis, Section 4 and Table 3 | `scripts/covariance.py` |
| Shadow, solar and Earth radiation pressure and drag, Section 5 and Table 4 | `scripts/perturbations.py` |
| Thermal recoil of the arrays, Section 5.3 | `scripts/thermal.py` |
| Figures 1 to 3 | `scripts/figures.py`, written to `figures/` |

The outcome table of Section 6, Table 5, is in `scripts/outcomes.py`, and `scripts/check_against_paper.py` compares every
number with the printed value.

## Running it

Python 3.9 or later with the packages in `requirements.txt`:

```
pip install -r requirements.txt
cd scripts
python check_against_paper.py
```

`check_against_paper.py` recomputes the numbers in the text, Tables 1 to 5 and Appendix A, compares each with the value
printed in the paper at the printed precision, and prints one PASS or FAIL line per number (198 in all, in about a
minute; the covariance analyses take most of it). Each of the other scripts prints its own section, and
`python figures.py` redraws the three figures in about a minute.

Everything runs offline from the stored states. `horizons_clipper.py` needs network access (it also runs in Google
Colab). Horizons serves its latest trajectory solution, so after further orbit determination the states it returns will
differ slightly from the stored ones; the scripts can be rerun on the new rows.

## Data

`data/horizons_states.csv` holds the three vectors the paper uses, all geocentric and at closest approach,
2026-12-03 20:15:20.7 TDB (20:14:11.5 UTC):

| Label | Contents |
|---|---|
| `clipper_perigee_icrf` | Europa Clipper (NAIF −159) position and velocity in the ICRF |
| `clipper_perigee_equator_of_date` | The same state in the Earth's true equator and node of date (Horizons `REF_PLANE='BODY'`); the declinations of the law are measured from this equator |
| `sun_at_perigee_icrf` | Position of the Sun in the ICRF |

They come from the Horizons trajectory segment `trj_260808-270123-dco2609230549-cruise019-pred-OD092`, a prediction from
orbit determination with tracking through 22 September 2026 (Horizons record revised 30 September 2026), retrieved in
October 2026.

`data/earth_flybys.csv` is the dataset of the first paper (columns described in the
[first repository](https://github.com/cerealrose/paper1-astg-flyby-foundation)). It supplies the couplings: the four-encounter
fit, the five-encounter refit of the second paper (Galileo II at its drag-corrected residual, +1.7628 mm/s, and Cassini
at −0.42 mm/s), the two-term angular law and the single-encounter couplings.

Spacecraft quantities follow the JPL press kit: arrays of 102 m<sup>2</sup> in total and a launch mass of 5800 kg. The 25 m<sup>2</sup> added
for the bus and high-gain antenna and the radiation-pressure coefficient of 1.2 are estimates, as the paper states.

## Companion papers

| Paper | Repository |
|---|---|
| Application of the ASTG-Model to Earth Flyby Encounters: Plausible Origins and Velocity Anomalies | [paper1-astg-flyby-foundation](https://github.com/cerealrose/paper1-astg-flyby-foundation) |
| A Drag Model for the Low-Perigee Galileo II Encounter and the Manoeuvre-Affected and Null Earth Gravity Assists | [paper2-astg-flyby-anomalies](https://github.com/cerealrose/paper2-astg-flyby-anomalies) |

## Licence and citation

Code is under the MIT licence and the data files under CC BY 4.0; see `LICENSE`. Citation details are in `CITATION.cff`.

Contact: P. A. Nesvinga, praise.nesvinga@nust.ac.zw
