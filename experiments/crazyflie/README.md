# Crazyflie reach-avoid experiment

Real-world pdSTL experiment (paper Experiment 3), laid out like the original
`irobot-crazyflie` codebase — but the pdSTL planner comes from this repo's
`src/` (no vendored copies) and the drone hardware comes from the lab's
`irobot` package.

```
generate_waypoints.py    offline: pdSTL planner -> components/opt_waypoints.py
main.py                  flight:  ros_sugar launcher for CrazyfliePlanning
components/
    environment_config.py  single source of truth: arena geometry (bounds,
                            obstacles, goal, start), planner config, and
                            build_environment()/build_planner() — everything
                            else imports from here instead of duplicating or
                            reaching into src/ directly
    calibration.py        hover-and-measure start offset, abort-if-too-large,
                           shift a plan to match the drone's real position
    config.py              ros_sugar component config (z_hold)
    crazyflie.py            flight component: calibrates at the start, flies
                             the (offset-corrected) plan, and replans from the
                             actual measured position at REPLAN_CHECKPOINTS
    flight_logger.py       commanded/actual CSV logging, auto-incremented runs
    opt_waypoints.py       generated nominal/warm-start WAYPOINTS list
                            (do not edit by hand — this is the *offline*
                            plan; calibration + mid-flight replanning correct
                            it against reality at flight time)
    logs/                  flight CSVs (gitignored)
```

## Why calibration + mid-flight replanning

An offline single-shot plan assumes the drone starts exactly where the plan
says and never drifts — real flights don't work that way (we saw a real
offset between the tracked position and the planned/simulated one last
time). So at flight time:

1. **Calibrate**: hover at the planned start, measure the real position for
   ~2s, and either abort (offset too large to trust) or shift the whole plan
   by the measured (dx, dy) so it's anchored to where the drone actually is.
2. **Replan mid-flight**: at each waypoint index in `REPLAN_CHECKPOINTS`
   (`environment_config.py`), read the live measured position and re-run the
   pdSTL optimizer (`Planner._optimize_window`) from there over the
   remaining horizon, splicing in a corrected tail. A full replan takes a
   few seconds on this hardware (measured: ~2.4s/300 iters, up to ~8s for a
   full 1000-iter convergence) — too slow to do before every waypoint
   (~0.7s apart), so this defaults to one checkpoint near the midpoint.
   Add a second index to `REPLAN_CHECKPOINTS` for two replans.

Old flight logs are not reused by any of this — this is a fresh experiment;
the logger's schema/behavior is unchanged, only its obstacle list now comes
from `environment_config.py` instead of being duplicated.

## Updating for a re-measured arena

Edit the geometry constants at the top of `components/environment_config.py`
(`FLIGHT_X_BOUNDS`, `FLIGHT_Y_BOUNDS`, `OBSTACLES`, `GOAL`, `START_XY`) —
one file, used by planning, flight, and logging alike. Current values are
placeholders carried over from the last measured arena; replace them before
the next flight.

## One-time setup

Planning needs only this repo (torch). Flying additionally needs, in the
flight environment:

```bash
pip install -e ~/Research/irobot     # hardware layer (CrazyflieBase)
# plus cflib and ros_sugar (ROS2), as before
```

**New dependency**: `components/crazyflie.py` now imports the pdSTL planner
(`torch`) at flight time too, for mid-flight replanning — it's no longer
decoupled through the generated `opt_waypoints.py` alone. The flight machine
needs `torch` installed alongside `cflib`/`ros_sugar`/`irobot`.

## Run flow (from this folder)

```bash
# 1. Plan (fan speed selects the process noise; --plot saves the before/after figure)
python generate_waypoints.py --fan 12 --plot

# 2. Fly: set the trial constants at the top of components/crazyflie.py
#    (USE_OPTIMISED, CONDITION, FAN_SPEED), then
python main.py
```

Logs land in `components/logs/` as `<condition>_fan<XX>_run<NN>_<ts>_{commanded,actual}.csv`;
crashed trials (including calibration aborts) are tagged `_CRASH`.

Planning is 2D at a fixed altitude (`Z_HEIGHT` in `environment_config.py`);
per-waypoint altitudes for future 3D flights go in `z_profile()` in
`generate_waypoints.py`.
