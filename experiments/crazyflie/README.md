# Crazyflie reach-avoid experiment

Real-world pdSTL experiment (paper Experiment 3), laid out like the original
`irobot-crazyflie` codebase — but the pdSTL planner comes from this repo's
`src/` (no vendored copies) and the drone hardware comes from the lab's
`irobot` package.

```
generate_waypoints.py   offline: pdSTL planner -> components/opt_waypoints.py
main.py                 flight:  ros_sugar launcher for CrazyfliePlanning
components/
    config.py           ros_sugar component config (z_hold)
    crazyflie.py        flight component (irobot CrazyflieBase + PositionHlCommander)
    flight_logger.py    commanded/actual CSV logging, auto-incremented runs
    opt_waypoints.py    generated WAYPOINTS list (do not edit by hand)
    logs/               flight CSVs (gitignored)
```

## One-time setup

Planning needs only this repo (torch). Flying additionally needs, in the
flight environment:

```bash
pip install -e ~/Research/irobot     # hardware layer (CrazyflieBase)
# plus cflib and ros_sugar (ROS2), as before
```

## Run flow (from this folder)

```bash
# 1. Plan (fan speed selects the process noise; --plot saves the before/after figure)
python generate_waypoints.py --fan 12 --plot

# 2. Fly: set the trial constants at the top of components/crazyflie.py
#    (USE_OPTIMISED, CONDITION, FAN_SPEED), then
python main.py
```

Logs land in `components/logs/` as `<condition>_fan<XX>_run<NN>_<ts>_{commanded,actual}.csv`;
crashed trials are tagged `_CRASH`.

Planning is 2D at a fixed altitude (`Z_HEIGHT`); per-waypoint altitudes for
future 3D flights go in `z_profile()` in `generate_waypoints.py`.
