# Crazyflie reach-avoid experiment

Real-world pdSTL experiment (paper Experiment 3). A Crazyflie flies from a start
to a goal through three obstacles under fan-induced disturbance, comparing a
**deterministic** nominal safe path against a **pdSTL**-optimised plan that
maximises the probabilistic satisfaction lower bound for a given fan level.

The pdSTL planner comes from this repo's `src/` (no vendored copies); the drone
hardware comes from the lab's `irobot` package.

## Layout

```
run.py                 entry point: plan (offline) | fly (hardware) | analyze (offline)
waypoint_planning.py   offline optimiser + before/after plotting
analyze_logs.py        post-flight: plot a logged run against its planned path
components/
    config.py          arena geometry, per-fan uncertainty, planner
                        hyperparameters, flight params — edit this one file
    crazyflie.py        flight component (ros_sugar): calibrate, fly the plan
                        start->finish, return, land. Defines CrazyflieConfig.
    calibration.py      hover-and-measure start offset, abort-if-too-large, shift
    flight_logger.py    commanded/actual CSV logging, auto-incremented runs
    logs/               flight CSVs (gitignored)
waypoints/              generated pdstl_fan<L>.json (one per fan level)
plots/                  fan<L>_comparison.png, <condition>_fan<XX>_run<NN>_actual.png
```

## Quickstart

Fly **deterministic before pdstl** — it needs no generated plan, so it
exercises calibrate → fly → log on the simplest path first.

**1. Install (planning only, any machine)**
```bash
cd experiments/crazyflie
pip install torch numpy matplotlib
```

**2. Smoke-test the planner (offline)**
```bash
python run.py plan --fan 2 --plot
```
Converges, prints `rho_before`/`rho_after`, writes `waypoints/pdstl_fan2.json`,
saves `plots/fan2_comparison.png`, and opens it in a window. If
`import pdstl`/`planning` fails, run from inside `experiments/crazyflie`
(paths resolve relative to `components/config.py`).

**3. Measure the arena**

Edit the geometry constants at the top of `components/config.py`:
`FLIGHT_X_BOUNDS`, `FLIGHT_Y_BOUNDS`, `OBSTACLES`, `GOAL`, `START_XY`,
`END_XY`, `SAFE_PATH_VIA_POINTS`. Every stage reads geometry from here.
Re-run step 2 and check the "Before" plot panel clears every obstacle.

**4. Install the hardware stack (flight machine)**
```bash
pip install torch numpy cflib
pip install -e <path-to-irobot-clone>   # e.g. pip install -e ~/Research/irobot
```
plus `ros_sugar` and ROS2. `torch` is required here too — `crazyflie.py`
imports `config.py`, which always imports `torch`/`pdstl`/`planning`.

Default radio URI is `radio://0/80/2M/E7E7E7E780`. To fly a different drone,
pass a `hw_config` when building `CrazyflieConfig` in `run.py`'s `_fly()`:
```python
from irobot.src.robots.crazyflie.config import CrazyflieConfig as CrazyflieHwConfig
CrazyflieConfig(..., hw_config=CrazyflieHwConfig(uri='radio://0/.../...'))
```

**5. Fly deterministic**
```bash
python run.py fly --condition deterministic --fan 6
```
`--fan` only tags the logs — the path itself doesn't depend on fan level.
Watch for the calibration offset printout; it aborts before takeoff if the
measured start is too far from `START_XY` (see [Calibration](#calibration)).

**6. Plan and fly pdstl**
```bash
python run.py plan --fan 12 --plot     # once per fan level
python run.py fly --condition pdstl --fan 12
```
`fly --condition pdstl --fan L` flies `waypoints/pdstl_fan<L>.json`, which
records the fan it was generated for and is checked on load. Generate a fan's
plan once, fly it as many times as you like. Repeat across fan levels (2, 6,
12, 16) to build out the comparison.

**7. Analyze**
```bash
python run.py analyze --condition pdstl --fan 12       # latest run
python run.py analyze --condition pdstl --fan 12 --run 3
python run.py analyze --all                             # every condition/fan pair
```
Writes `plots/<condition>_fan<XX>_run<NN>_actual.png`: actual flown path,
commanded waypoints, and planned path on one arena drawing; unsafe samples
(inside an obstacle) marked red.

## Logging

`fly` writes `components/logs/<condition>_fan<XX>_run<NN>_<ts>_{commanded,actual}.csv`
— one row per commanded waypoint, and 10 Hz sampled real position. Run number
auto-increments per `(condition, fan)`; crashed trials (including calibration
aborts) are tagged `_CRASH`. The return-to-start leg after each trial flies
but isn't logged.

## Per-fan uncertainty

Each fan level has its own initial belief covariance Σ0 (`SIGMA0_PER_FAN` in
`config.py`):

| Fan | Σ0 (m²) | Source |
|----:|--------:|--------|
| 2   | 0.001   | paper Setting 1 |
| 6   | 0.006   | paper Setting 2 |
| 12  | 0.020   | paper Setting 3 |
| 16  | 0.050   | uncalibrated extrapolation (no paper source) |

Belief grows along the path as `Σ_t = Σ0 + t·Q_STD²`, `Q_STD` shared across
fans (small relative to Σ0, so per-fan differences dominate).

## Optimiser knobs

All in `PLANNER_CONFIG` in `config.py`: paper-faithful (`w_phi`, `w_u`=λ,
`alpha`), heuristic shaping (`w_dist`, `w_obs`, `w_visit`, `obs_margin` —
speeds convergence, not in the paper's objective, set to 0 for the clean
objective), and numerical (`lr`, `max_iters`, …). `scale` controls β
log-sum-exp smoothing of the STL min/max: `-1` = exact (default), `>0` =
smooth.

## Calibration

An offline plan assumes an exact start position. Real flights drift, so at
flight time the drone hovers at the planned start, measures its real position
for ~2 s, and either aborts (offset too large) or shifts the whole plan by the
measured offset — a one-time pre-flight step.

There is no mid-flight replanning: the plan flies start-to-finish as given, at
the planned `U_MAX`, so actual flight matches the belief model's timing.

## Scaling to 3D

Planning is currently 2D at a fixed altitude (`Z_HEIGHT`). The path to 3D:

- `z_profile()` in `waypoint_planning.py` — currently constant, would carry a
  real per-waypoint altitude.
- `OBSTACLES` already carry a `height` field, unused by the 2D collision logic.
- `SingleIntegrator` in `src/planning/dynamics.py` — `eye(2)` → `eye(3)`.
- `Rectangular{Goal,Obstacle}` predicates in `src/planning/environment.py` and
  the planner's `[:2]` slices in `src/planning/planner.py` — extend to 3D.
- `src/pdstl/` operators are already dimension-agnostic; no change needed there.

So a 3D experiment is a localised change to the dynamics/environment/planner
slicing, not a rewrite — the belief/STL machinery underneath is unaffected.

## Troubleshooting

- **matplotlib `CXXABI_... not found`** on `--plot`: conda libstdc++ mismatch.
  `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" python run.py plan --fan 2 --plot`.
- **`No optimised waypoints for fan L`**: generate it first with
  `python run.py plan --fan L`.
