# Crazyflie reach-avoid experiment

Real-world pdSTL experiment (paper Experiment 3), laid out like the original
`irobot-crazyflie` codebase — but the pdSTL planner comes from this repo's
`src/` (no vendored copies) and the drone hardware comes from the lab's
`irobot` package.

```
generate_waypoints.py    offline: pdSTL planner -> components/opt_waypoints.py
main.py                  flight:  CLI entry point, launches CrazyfliePlanning
components/
    environment_config.py  single source of truth: arena geometry (bounds,
                            obstacles, goal, start), the deterministic safe
                            path's via-points, planner config, and
                            build_environment()/build_planner() — everything
                            else imports from here instead of duplicating or
                            reaching into src/ directly
    calibration.py        hover-and-measure start offset, abort-if-too-large,
                           shift a plan to match the drone's real position
    config.py              CrazyflieConfig — trial settings (condition, fan
                            speed) live here, set from main.py's CLI args
    crazyflie.py            flight component: calibrates at the start, flies
                             the (offset-corrected) plan, and replans from the
                             actual measured position at REPLAN_CHECKPOINTS
    flight_logger.py       commanded/actual CSV logging, auto-incremented runs
    opt_waypoints.py       generated pdSTL-optimised WAYPOINTS list
                            (do not edit by hand — regenerate with
                            generate_waypoints.py)
    logs/                  flight CSVs (gitignored)
```

## Setup (works on any machine — no hardware needed for this part)

**Prerequisites**: Python 3.10+, and this repo's `src/` importable (that's
handled automatically — `environment_config.py` adds `<repo>/src` to
`sys.path`, no install step needed for `pdstl`/`planning` themselves).

Install the planning dependencies (not fully covered by the repo's
`requirements.txt`, which is missing `torch`/`numpy`):
```bash
pip install torch numpy matplotlib
```

**Smoke test** — confirms the install works, no drone or ROS needed:
```bash
cd experiments/crazyflie
python generate_waypoints.py --fan 2 --plot
```
This should converge and write `components/opt_waypoints.py` plus
`waypoints_comparison.png` (before/after path figure). If it errors on
`import torch`/`numpy`, the pip install above didn't take; if it errors on
`import pdstl`/`planning`, you're not running from inside
`experiments/crazyflie` (the path resolution is relative to this file).

### Hardware setup (only needed to actually fly)

On top of the above, install:
```bash
pip install cflib
pip install -e <path-to-irobot-clone>   # e.g. pip install -e ~/Research/irobot
```
Plus `ros_sugar` and a working ROS2 install (not pip-installable; follow your
lab's ROS2 setup). `components/crazyflie.py` imports the pdSTL planner
(`torch`) at flight time too, for mid-flight replanning — it's no longer
decoupled through the generated `opt_waypoints.py` alone, so the flight
machine needs the full set above (`torch`, `cflib`, `ros_sugar`, `irobot`).

## Why calibration + mid-flight replanning

An offline single-shot plan assumes the drone starts exactly where the plan
says and never drifts — real flights don't work that way (we saw a real
offset between the tracked position and the planned/simulated one). So at
flight time:

1. **Calibrate**: hover at the planned start, measure the real position for
   ~2s, and either abort (offset too large to trust) or shift the whole plan
   by the measured (dx, dy) so it's anchored to where the drone actually is.
2. **Replan mid-flight**: at each waypoint index in `REPLAN_CHECKPOINTS`
   (`environment_config.py`), read the live measured position and re-run the
   pdSTL optimizer (`Planner._optimize_window`) from there over the
   remaining horizon, splicing in a corrected tail. A full replan takes a
   few seconds on real hardware (measured: ~2.4s/300 iters, up to ~8s for a
   full 1000-iter convergence) — too slow to do before every waypoint
   (~0.7s apart), so this defaults to one checkpoint near the midpoint.
   Add a second index to `REPLAN_CHECKPOINTS` for two replans.

## The deterministic condition

`nominal_safe_waypoints()` in `environment_config.py` is a path computed to
safely clear all three obstacles **with no disturbance/wind modelled** —
via-points (`SAFE_PATH_VIA_POINTS`) chosen by hand to route around each
obstacle, connected by a monotone (PCHIP) spline. The risk this condition is
meant to surface comes from *flying* that nominally-safe path under real fan
noise, not from the path itself cutting through anything. It also serves as
the pdSTL optimizer's warm start.

Old flight logs are not reused by any of this — this is a fresh experiment;
the logger's schema/behavior is unchanged, only its obstacle list comes from
`environment_config.py` instead of being duplicated.

## Updating for a re-measured arena

Edit the geometry constants at the top of `components/environment_config.py`
(`FLIGHT_X_BOUNDS`, `FLIGHT_Y_BOUNDS`, `OBSTACLES`, `GOAL`, `START_XY`,
`END_XY`, `SAFE_PATH_VIA_POINTS`) — one file, used by planning, flight, and
logging alike.

## Run flow (from this folder)

```bash
# 1. Plan (fan speed selects the process noise; --plot saves the before/after figure)
python generate_waypoints.py --fan 12 --plot

# 2. Fly — trial settings are CLI args, no file editing required:
python main.py --condition pdstl --fan 12            # fly the pdSTL-optimised plan
python main.py --condition deterministic --fan 6      # fly the nominal safe path
```

Logs land in `components/logs/` as `<condition>_fan<XX>_run<NN>_<ts>_{commanded,actual}.csv`;
crashed trials (including calibration aborts) are tagged `_CRASH`. The
return-to-start leg after each trial is flown but not logged, so back-to-back
runs don't need any manual reset between them.

Planning is 2D at a fixed altitude (`Z_HEIGHT` in `environment_config.py`);
per-waypoint altitudes for future 3D flights go in `z_profile()` in
`generate_waypoints.py`.

## Noise calibration

`Q_STD_PER_FAN` in `environment_config.py` maps fan settings to process
noise. Three of its four values (`2, 6, 12`) are verified against the paper's
documented calibration; the fourth (`16`) has no paper source. A separate,
more rigorous calibration effort exists in `~/Research/crazyflie-stl-experiments`
(`cfstl/calibrate_noise.py`, real tracking-residual data for fan levels
`00/06/12/18`) but uses a different fan-level scheme that hasn't been
reconciled with this experiment's — not wired in here.
