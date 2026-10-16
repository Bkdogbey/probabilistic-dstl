# Crazyflie figure-8 test

Checks that the drone tracks a figure-8 with velocity commands before any
planning is added. Metres, seconds, m/s; Lighthouse world frame.

| File | Purpose |
| --- | --- |
| `config.yml` | Radio URI, arena, altitude and figure-8 size. Edit this. |
| `config.py` | Typed defaults for everything not set in `config.yml` |
| `paths.py` | The figure-8 and the timed reference along it |
| `arena.py` | The task (workspace, obstacles, goal) and its robustness |
| `tracking.py` | Settle check, velocity command, safety checks, control loop |
| `flight.py` | ROS Sugar component: connect, takeoff, track, land, CSV |
| `run.py` | Launcher |
| `analyze.py` | `--preview` of the path; one flight; `--all` flights in a table |
| `model.py` | Fit the planner's motion noise from completed flights; `model.json` |

The reference moves along the figure-8 at `path.speed`. Each step at 20 Hz
sends `v = v_ref + gain * (r(t) - position)`, clamped to `control.v_max`,
with `send_velocity_world_setpoint`. The flight lands if the state is stale,
the drone leaves the workspace by more than `flight.fence_margin`, it falls
more than `control.max_error` behind the reference, or it cannot settle at
the end point.

## Change the setup

Edit `config.yml`: `radio_uri`, `workspace`, `altitude` and the figure-8
(`centre`, `width`, `height`, `speed`). A path that leaves the workspace is
rejected before connecting, and unknown keys are an error. Defaults for gains
and safety limits are in `config.py`; override them in `config.yml` under
`control:` or `flight:`.

## The task

`config.yml` also holds the task the figure-8 is scored against: stay in the
workspace, avoid the `obstacles`, and reach the `goal`. The figure-8 is the
deterministic path. The specification is built with `planning.environment`
from `src/`, the same one the pdSTL planner uses. Obstacles grow and the
workspace shrinks by `airframe_radius`, so the checks are on the drone's
centre.

`analyze.py` prints the robustness (> 0 means the task is satisfied) and
the closest approach to a wall or obstacle, for the planned path
(`--preview`) and for each flight. The task never blocks or aborts a flight.
The boxes in `config.yml` must match the physical layout; measure them and
update the file before trusting the scores.

## Run

```bash
conda activate ros_env
source ~/sugar_ws/install/setup.bash
cd ~/Research/probabilistic-dstl/experiments/crazyflie
export PYTHONPATH=~/Research/irobot

python analyze.py --preview   # path length, time, task score; no hardware
python ~/Research/irobot/irobot/src/robots/crazyflie/examples/check_positioning.py
python run.py                 # drone anywhere in the arena
python analyze.py             # latest flights/*.csv
python analyze.py --all       # every flight, and the task success rate
python model.py               # fit q_std from completed flights
```

Place the drone anywhere in the arena. It takes off where it sits and flies a
straight line to the bottom tip at `flight.transit_speed`, then follows the
figure-8; only the figure-8 is recorded. If that line would cross an
obstacle, it refuses to take off; move the drone. Use propeller guards and keep a hand on
Ctrl-C, which lands. After landing, press Ctrl-C again to exit the idle
component. Each flight is saved in `flights/` as a CSV and a JSON copy of the
config used.

Tests need no hardware: `pytest experiments/crazyflie/tests -q`.

## Baseline and model

Fly the figure-8 repeatedly under the same conditions. `analyze.py --all`
lists each flight (outcome, tracking error, robustness, closest approach) and
the fraction of completed flights that satisfy the task: the deterministic
baseline.

`model.py` fits the planner's single-integrator model,
`x[k+1] = x[k] + u[k] dt + w`, `w ~ N(0, q_std^2 I)`, from the completed
flights. Each step's residual is where the drone went minus where the
commanded velocity would have taken it. It prints q_std per flight (they
should agree) and saves `model.json` with q_std, Q, the bias, the start
covariance, and `track_cov`, the covariance of the closed-loop tracking error.
Aborted flights are left out.

## If it will not take off

- `Too many packets lost`: wrong `radio_uri` in `config.yml`, or the drone is
  off or out of range.
- `position never settled`: the Lighthouse estimate keeps jumping. Run
  `check_positioning.py`; the drone needs to see both base stations.
- The traceback after Ctrl-C comes from ROS Sugar shutting down while its
  executor is still spinning; it is harmless.
