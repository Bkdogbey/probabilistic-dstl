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

**Fly** (`flight.py`, `telemetry.py`, `flight_log.py`, `component.py`).
The ROS Sugar `FlightComponent` runs `flight.run_flight` in these steps:

1. Connect with irobot `CrazyflieBase`, start the experiment's own log
   blocks and wait for the first fresh snapshot.
2. Reset the Kalman filter and wait for a stable position estimate.
3. Take off with the high-level commander. If the drone is not within
   `home_tolerance` of the start, it climbs to `transit_z` first.
4. Fly to the start and settle there.
5. Stream the trajectory as 20 Hz position setpoints, aborting if the
   geofence or the radio link is lost.
6. Land, stop telemetry, disarm and disconnect.

Any error or Ctrl-C lands the drone if it is airborne, then disarms and
disconnects.

**Telemetry contract** (`telemetry.py`):

- **Log blocks:** two experiment-owned blocks, both at
  `telemetry.period_ms`:
  - `state` = `stateEstimate.x/y/vx/vy` and `ctrltarget.x/y`
  - `alt` = `stateEstimate.z/vz`, `ctrltarget.z` and `pm.vbat`
- **One row per packet:** every row is one device packet, with its
  24-bit device timestamp and host receive time. Nothing cached is
  logged twice.
- **Fresh data for safety:** every safety decision (geofence, start gate,
  positioning, landing height) uses a *snapshot*:
  - x/y come from the newest `state` packet and z from the newest `alt`
    packet
  - both must be fresher than `telemetry.stale_s` and sampled within 1.5
    periods of each other

  Otherwise the flight aborts and lands.
- **No disk I/O in callbacks:** callbacks only update the latest packet
  and `put_nowait` the row into a bounded queue. One writer thread writes
  and flushes each batch. A full queue drops rows and counts them.
- **Command timing:** each streamed command logs its reference time
  `t_ref`, send time and lateness. A command later than
  `flight.max_lateness_s` aborts the trajectory instead of being sent
  late, so there is never a catch-up burst.

Every attempt writes `logs/<UTC>_<name>_<method>_<hex>/` containing:

- `config.yaml`, `trajectory.csv` and `plan.json`: what was flown
- `commands.csv`: `t_send, t_ref, k, x, y, z, yaw, lateness_s, phase`
- `state.csv`: `device_ms, t_recv, x, y, vx, vy, target_x, target_y,
  phase`
- `alt.csv`: `device_ms, t_recv, z, vz, target_z, vbat, phase`
- `events.csv`: phase changes, stale landings, aborts and log-block
  errors
- `summary.json`, including a `data` block with:
  - rows and dropped rows per file
  - writer errors
  - queue depth and write lag
  - device-time gaps and rate per block
  - command lateness (p50, p99, max)
- `plan.png` and `flight.png`

**Analyze** (`analysis.py`, `plots.py`). For each run, analyze reports:

- the plan's probability
- tracking error: the estimate against the **active controller target
  from the same packet** (device clock, no host alignment). This
  measures flight performance; it is not process noise.
- data quality: device-time gaps, command lateness, dropped rows
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

**Planning and analysis:** the repo's `pdstl` environment (torch and
matplotlib). pdSTL plans need torch, so plan there; flights reuse the
saved plan as long as the config is unchanged.

**Flight:** the conda ROS 2 Humble environment `ros_env`, with ROS Sugar
built in its own workspace `~/sugar_ws`. One-time setup (already done on
this machine):

```bash
conda activate ros_env
pip install -e ~/Research/irobot          # also installs trossen_arm
git clone https://github.com/automatika-robotics/sugarcoat ~/sugar_ws/src/sugarcoat
cd ~/sugar_ws && colcon build --packages-select automatika_ros_sugar \
    --cmake-args -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python -DPython_FIND_STRATEGY=LOCATION
```

Every flight session:

```bash
conda activate ros_env
source ~/sugar_ws/install/setup.bash
python experiments/crazyflie/run.py fly --dry-run --ros   # rehearse the ROS path
python experiments/crazyflie/run.py fly                   # real flight
```

`fly` checks for ROS Sugar, cflib and irobot before planning or
prompting, and says what is missing.

Under ROS Sugar the mission runs in its own thread, so a shutdown can
reach it. On Ctrl-C:

1. The flight stops and lands.
2. `summary.json` records `interrupted`.
3. The process waits for the landing before exiting. A second Ctrl-C
   while landing is ignored.

ROS Sugar itself prints an `RCLError ... context is not valid` traceback
from its executor threads during shutdown. It is harmless; the launcher
still reports `ALL COMPONENTS EXITED SUCCESSFULLY`.

## Tests

```bash
PYTHONPATH=src pytest experiments/crazyflie/tests -q
```
