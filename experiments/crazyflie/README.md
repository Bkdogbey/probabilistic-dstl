# Crazyflie ring course

This folder runs a Crazyflie through a ring course in three stages:
`plan`, `fly` and `analyze`. All three read one config file,
`config.yaml`, and share a single entry point, `run.py`.

```text
config.yaml ──► plan ──► plans/<name>/<method>/     trajectory.csv, plan.json, plan.png
                  │        (reused while its signature matches the config)
                  ▼
                fly ───► logs/<run_id>/             config, plan, commands, telemetry,
                  │        ROS Sugar component       summary.json, flight.png
                  │        → irobot Crazyflie
                  ▼
              analyze ─► per-run metrics + figure; --all → runs.csv, repeats.png
```

## Commands

```bash
python experiments/crazyflie/run.py plan                       # plan + figure, no flight
python experiments/crazyflie/run.py fly --dry-run              # simulated drone
python experiments/crazyflie/run.py fly                        # real flight: shows plan, asks y/N
python experiments/crazyflie/run.py analyze                    # latest run
python experiments/crazyflie/run.py analyze --all              # every run + repeatability
```

`--method s_path|pdstl` on `plan` or `fly` overrides `planning.method` for
that command. `fly --replan` ignores the saved plan, and `fly --yes` skips
the prompt.

## Stages

**Plan** (`planners.py`). Both planners fly the same task: from
`planning.start`, through the enabled rings in order, to `planning.goal`
at `planning.height`, avoiding any enabled obstacles.

- `s_path` is a deterministic S along the start→goal line, flown at
  constant speed.
- `pdstl` uses the planner in `src/`. `pdstl_scenario` maps the arena
  onto a reach-avoid problem:
  - the bounds become the workspace
  - obstacles are inflated by their clearance
  - each ring becomes the square inside its disc, with its own ordered
    time window
  - the goal becomes a square around the goal point

  The solve starts from a straight line through the rings. The flown path
  is the planned mean at `k·dt`. `plan.json` keeps the probability
  interval, the α status, a Monte Carlo check and the per-step
  covariance; the figures draw that covariance as 95 % ellipses.

`trajectory.check` validates every plan before it is saved or flown. It
checks timing, bounds, speed and the enabled environment constraints,
using `environment.py`. Each saved plan stores a **signature** of
`planning`, `bounds`, `environment` and `flight.max_speed`. `fly` reuses
the saved plan only while that signature still matches the config;
otherwise it plans again, so a flight can never use a stale plan.

**Fly** (`flight.py`, `component.py`, `flight_log.py`). The ROS Sugar
`FlightComponent` runs `flight.run_flight` in these steps:

1. Connect with irobot `CrazyflieBase`.
2. Reset the Kalman filter and wait for a stable position estimate.
3. Take off with the high-level commander. If the drone is not within
   `home_tolerance` of the start, it climbs to `transit_z` first.
4. Fly to the start and settle there.
5. Stream the trajectory as 20 Hz position setpoints, aborting if the
   geofence or the radio link is lost.
6. Land, disarm and disconnect.

Any error or Ctrl-C lands the drone if it is airborne, then disarms and
disconnects. Every attempt writes `logs/<UTC>_<name>_<method>_<hex>/`
containing:

- `config.yaml`, `trajectory.csv` and `plan.json`: what was flown
- `commands.csv` and `telemetry.csv`: flushed on every row, so partial
  runs keep their data
- `summary.json`: outcome, last phase, error, tracking error, ring
  passage and the plan signature
- `plan.png` and `flight.png`

**Analyze** (`analysis.py`, `plots.py`). For each run, analyze reports:

- the plan's probability
- tracking error: the estimated position against the commanded setpoint
  at the same time
- the closest flown approach to each ring

The flight figure overlays the flown path on the plan and its
ellipses. `--all` writes `logs/runs.csv`, with one row per attempt
including method and P(φ). It also writes `logs/repeats.png`, which
overlays every completed run of the same plan.

## Config

All values live in `config.yaml`. The arena, radio and flight values
follow `experiments/RA-L`. Under `environment`, each group is switched on
with `enabled`, and the figures draw disabled groups dashed:

- `obstacles`: the RA-L figure-eight boxes. They are off for the ring
  course.
- `rings`: the three rings, in flight order. The centres and radii are
  **placeholders** on the nominal S. Enter the measured values, then
  enable them.

The pdSTL noise settings `q_std` and `x0_var` are **placeholders** until
tracking is calibrated from repeated flights (`analyze --all`). Until
then, P(φ) is only indicative. The prompt still shows plans whose lower
bound is below α, flagged as such.

## Setup

- **Planning and analysis:** the repo's `pdstl` environment (torch,
  matplotlib).
- **Flight:** one Python environment with ROS 2, `ros_sugar`, `cflib`, numpy, PyYAML, matplotlib,
  `irobot` and `trossen_arm`. Install `irobot` with `pip install -e` from
  `Bkdogbey/irobot`, branch `crazyflie`; irobot's `__init__` imports
  `trossen_arm`, so that package is needed too. Saved plans in `plans/`
  let the flight machine fly without torch, as long as the config is
  unchanged.

## Tests

```bash
PYTHONPATH=src pytest experiments/crazyflie/tests -q
```
