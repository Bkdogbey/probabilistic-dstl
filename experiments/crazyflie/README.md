# Crazyflie reach-avoid experiment

Real-world pdSTL experiment (paper Experiment 3). A Crazyflie flies from a start
to a goal through three obstacles under fan-induced disturbance. Two conditions
are compared: a **deterministic** nominal safe path, and a **pdSTL**-optimised
plan that maximises the probabilistic satisfaction lower bound for a given fan
level.

The pdSTL planner comes from this repo's `src/` (no vendored copies); the drone
hardware comes from the lab's `irobot` package.

## Layout

```
run.py                 SINGLE entry point:  plan (offline) | fly (hardware)
waypoint_planning.py   offline optimiser + before/after plotting
components/
    config.py          SINGLE config: arena geometry, per-fan uncertainty,
                        planner hyperparameters, flight params, factories.
                        Edit this one file to change anything. No hardware/ROS
                        import, so the offline `plan` path stays lightweight.
    crazyflie.py        flight component (ros_sugar): calibrate at start, fly the
                        plan start->finish, return, land. Also defines the
                        CrazyflieConfig trial dataclass.
    calibration.py      hover-and-measure start offset, abort-if-too-large, shift
    flight_logger.py    commanded/actual CSV logging, auto-incremented runs
    logs/               flight CSVs (gitignored)
waypoints/              generated pdstl_fan<L>.json (one per fan level)
plots/                  generated fan<L>_comparison.png (one per fan level)
```

Everything you configure lives in **`components/config.py`**. Trial choices
(condition, fan) are CLI args to `run.py` — no file editing per run.

## Run flow

```bash
cd experiments/crazyflie

# 1. Plan (offline, no hardware). Fan level selects the initial belief
#    covariance Σ0; --plot writes plots/fan<L>_comparison.png.
python run.py plan --fan 12 --plot

# 2. Fly (needs hardware + ROS). No replanning — the plan is flown as given.
python run.py fly --condition pdstl --fan 12           # fly the optimised plan
python run.py fly --condition deterministic --fan 6    # fly the nominal safe path
```

`fly --condition pdstl --fan L` flies `waypoints/pdstl_fan<L>.json`, which records
the fan it was generated for and is checked on load — so you can't accidentally
fly a plan optimised for a different fan. Generate each fan's plan once; fly any
of them later without re-planning.

Logs land in `components/logs/` as
`<condition>_fan<XX>_run<NN>_<ts>_{commanded,actual}.csv`; crashed trials
(including calibration aborts) are tagged `_CRASH`. The return-to-start leg after
each trial is flown but not logged, so back-to-back runs need no manual reset.

## Setup

**Planning only** (works on any machine — no hardware):
```bash
pip install torch numpy matplotlib
```
Smoke test:
```bash
python run.py plan --fan 2 --plot
```
This converges, writes `waypoints/pdstl_fan2.json` and `plots/fan2_comparison.png`.
If `import pdstl`/`planning` fails, you're not running from inside
`experiments/crazyflie` (path resolution is relative to `components/config.py`).

**Flying** additionally needs the hardware stack (`fly` lazy-imports these, so
`plan` never requires them):
```bash
pip install cflib
pip install -e <path-to-irobot-clone>   # e.g. pip install -e ~/Research/irobot
```
plus `ros_sugar` and a working ROS2 install (follow your lab's ROS2 setup). Note
the flight machine **no longer needs torch** — replanning was removed, so the
flight path only loads pre-generated JSON waypoints.

## Per-fan uncertainty (the covariance model)

Each fan level has its own **initial belief covariance Σ0** (`SIGMA0_PER_FAN` in
`config.py`), so every fan's optimisation and plot reflect that fan's real
uncertainty:

| Fan | Σ0 (m²) | Source |
|----:|--------:|--------|
| 2   | 0.001   | paper Setting 1 |
| 6   | 0.006   | paper Setting 2 |
| 12  | 0.020   | paper Setting 3 |
| 16  | 0.050   | uncalibrated extrapolation (no paper source) |

The belief grows along the path as `Σ_t = Σ0 + t·Q_STD²`, where `Q_STD` is a small
shared per-step process noise (modest vs Σ0, so the per-fan Σ0 differences
dominate the plots). Bump `Q_STD` for more visible growth, or calibrate both from
tracking residuals.

This replaces the previous behaviour, where every fan used a single hardcoded
Σ0 = 0.01 and all plots looked identical.

## Optimiser knobs

Every planner hyperparameter is in one place — `PLANNER_CONFIG` in `config.py` —
with each value labelled paper-faithful (`w_phi`, `w_u`=λ, `alpha`), heuristic
shaping (`w_dist`, `w_obs`, `w_visit`, `obs_margin` — potential fields that speed
convergence but aren't in the paper's objective; set to 0 for the clean paper
objective), or numerical (`lr`, `max_iters`, …). The `scale` knob controls β
log-sum-exp smoothing of the STL min/max: `-1` = exact (default, current
behaviour), `>0` = smooth (the paper's differentiable form).

## Why calibration (but not replanning)

An offline plan assumes the drone starts exactly where the plan says. Real flights
drift, so at flight time the drone hovers at the planned start, measures its real
position for ~2 s, and either aborts (offset too large to trust) or shifts the
whole plan by the measured (dx, dy). This is a **one-time pre-flight** step.

There is **no mid-flight replanning** — the plan is flown start-to-finish as
given, so the drone moves continuously without pausing mid-arena to re-optimise.
Flight speed is set to the planned `U_MAX` so actual flight matches the belief
model's timing.

## Updating for a re-measured arena

Edit the geometry constants at the top of `components/config.py`
(`FLIGHT_*_BOUNDS`, `OBSTACLES`, `GOAL`, `START_XY`, `END_XY`,
`SAFE_PATH_VIA_POINTS`) — one file, used by planning, flight, and logging alike.

## Scaling to 3D (future)

Planning is 2D at a fixed altitude (`Z_HEIGHT`); per-waypoint altitudes go in
`z_profile()` in `waypoint_planning.py`, and `OBSTACLES` already carry a `height`
field. A full 3D experiment is a localised change: `SingleIntegrator`
(`eye(2)`→`eye(3)`) in `src/planning/dynamics.py`, the `Rectangular{Goal,Obstacle}`
predicates in `src/planning/environment.py`, and the planner's `[:2]` slices in
`src/planning/planner.py`. The pdSTL operators (`src/pdstl/`) are already
dimension-agnostic.

## Troubleshooting

- **matplotlib `CXXABI_... not found`** on `--plot`: a conda libstdc++ mismatch.
  Run with the env's own lib: `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" python run.py plan --fan 2 --plot`.
- **`No optimised waypoints for fan L`** on `fly --condition pdstl`: generate that
  fan's plan first with `python run.py plan --fan L`.
