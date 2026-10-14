# Crazyflie pdSTL experiment

## Setup

Run every command from this directory (`experiments/crazyflie`).

1. **Python environment.** Use the ROS environment for everything: planning,
   dry runs, flights, fitting and analysis. Once, install what it lacks:

   ```bash
   conda activate ros_env
   pip install torch --index-url https://download.pytorch.org/whl/cpu   # pdSTL core
   pip install cflib
   pip install -e ~/Research/irobot --no-deps   # --no-deps skips trossen-arm
   ```

   Then, in every new terminal:

   ```bash
   conda activate ros_env
   source ~/sugar_ws/install/setup.bash
   cd ~/Research/probabilistic-dstl/experiments/crazyflie
   ```

   `fly` checks for ROS Sugar, cflib and irobot before connecting, and says
   how to install whichever is missing.
2. **Local configuration.** `config.yml` is not tracked by git. Create it from
   the template, then enter the measured arena:

   ```bash
   cp config.example.yml config.yml
   ```

   The template's geometry is a placeholder from the RA-L arena. Before the
   first real flight, check `workspace`, `radio_uri`, `altitude` and
   `start`. Identification and evaluation flights stay locked until
   `measured: true`.

## First flight

```bash
python run.py plan --condition deterministic
python run.py fly --purpose pilot --dry-run
python run.py fly --purpose pilot                # ROS env sourced
python run.py analyze --trial latest
```

No paths or ids to type. `fly` uses the newest plan for the condition (and
fan) made under the current `config.yml`; `analyze --trial` takes `latest`.
To pick another, pass an id (`--plan deterministic_reference_20261014T174111Z`)
or a unique part of a trial id (`--trial 174113`).

1. **Generate the plan** and inspect the path figure saved next to it
   (`plans/<plan id>.png`).
2. **Run the simulated flight** (`--dry-run`: no radio, no ROS).
3. **Fly the same plan on hardware.** Before asking for confirmation, `fly`
   prints:
   - the plan figure path
   - duration, altitude, start and maximum speed
4. **Inspect the trial.** `analyze --trial` prints:
   - command lateness and state-packet gaps
   - the start state and battery

   It also saves the commanded-versus-measured figure as
   `logs/<trial id>/flight.png`.

## Overview

The drone flies horizontal velocity commands at a fixed altitude.

1. Deterministic flights set up execution and collect identification data.
2. A tracking model is fitted and checked on held-out flights.
3. `mean_model` plans (STL robustness of the predicted mean) are compared with
   `pdstl` plans (pdSTL lower bound). Both use the same task, model,
   initialization, limits and optimization budget.

```text
config.yml ─▶ plan (deterministic) ─▶ fly (identification) ─▶ fit ─▶ reports/
                                                                       │
          analyze ◀── fly (evaluation) ◀── plan (mean_model | pdstl) ◀─┘
```

| File | Role |
|---|---|
| `config.example.yml`, `config.py` | Template for the local `config.yml`; all inputs as one typed `Config` |
| `run.py` | Entry point: `plan`, `fly`, `fit`, `analyze` |
| `planner.py` | Task geometry and pdSTL spec, reference/excitation commands, both planners, plan files |
| `flight.py` | Preflight, start gate, 20 Hz velocity streaming, abort, landing, ROS Sugar launch, `SimDrone` |
| `logger.py` | `logs/<trial>/commands.csv`, `state.csv`, `metadata.json`; loading and time alignment |
| `tracking_model.py` | `TrackingModel` (first-order velocity lag with fan bias `c_f` and noise `Q_f`), fit, validation |
| `analysis.py` | Mission outcomes, success with Clopper–Pearson CIs, figures, single-trial check |

`planner.py` is not named `planning.py` because that would shadow the core
`planning` package in `src/`.

## All commands

```bash
python run.py plan --condition deterministic                      # fixed reference
python run.py plan --condition deterministic --sequence steps_x   # identification excitation
python run.py fly --purpose pilot --dry-run                       # newest plan, simulated
python run.py fly --purpose identification --fan off              # newest deterministic plan
python run.py analyze --trial latest                              # timing + commanded vs measured
python run.py fit                                                 # reports/model_<UTC>.json + .png
python run.py plan --condition pdstl --fan off                    # newest report by default
python run.py plan --condition mean_model --fan off
python run.py fly --condition pdstl --purpose evaluation --fan off
python run.py analyze                                             # results/summary.{csv,md}, paths.png
pytest tests -q                                                   # offline checks, SimDrone
```

`--dry-run` flies the simulated drone with no radio and no ROS. Its trials
are marked `dry_run` and never used by `fit` or `analyze`.

## Rules that keep the data honest

- **Measure first.** Identification and evaluation flights refuse to fly
  until `measured: true` and the chosen fan's setting, position and direction
  are filled in. Pilot flights are always allowed.
- **Plans are frozen.** Each plan is a new file in `plans/` that records:
  - the config hash and model id/hash
  - the predicted μ/Σ and per-step safe and goal probabilities
  - its score: hard pdSTL lower bound, or mean robustness

  `fly` refuses a plan made with a different config, an infeasible one, or one
  that missed its threshold (`planner.alpha`, `mean_model.min_robustness`).
- **Every attempt is recorded.** `metadata.json` is written before
  connecting. If the flight aborts or fails, it still gets the outcome, the
  abort kind and phase, and whatever data exists.
- **Accounting.**
  - A `TaskViolation` abort (geofence or altitude) counts as a failed flight.
  - A `DataFault` (stale state, lost link, late command, start gate),
    errors and interruptions are excluded, with the reason listed.
- **Held-out data.** Within each fan, every `model.holdout_every`-th
  identification flight is held out of fitting and used only for validation.
- **Model bound ≠ flight outcome.** A plan's pdSTL score is reported next to
  the empirical success rate, never merged with it.

## Execution details

- **Aborts land.** On any abort the executor calls
  `send_notify_setpoint_stop`, then lands with the high-level commander. It
  does not just stop transmitting. On firmware without a supervisor, check
  what happens when communication is lost.
- **Command timing.** The mission starts once the drone is within
  `start_tolerance` of the start and below `start_speed`, after `settle`
  seconds and a recheck. Command `k` is due at `t0 + k/20 s`. Each 0.2 s plan
  step is sent 4 times, rotated into the body frame using the measured yaw.
  A command more than `max_lateness_s` late aborts the flight.
- **State log.** One 20 Hz block carries `x y z vx vy yaw`, so a state sample
  is never stitched together from two packets. The device clock is mapped to
  host time by the smallest receive delay.

## Milestones

1. **A repeatable deterministic reference flight with aligned command and
   state logs.** Check it with `analyze --trial`: lateness, packet gaps, the
   start state, and commanded vs measured velocity.
2. Repeated reference and excitation flights per fan, then `fit`. Check
   held-out error vs spread, coverage (39.3/86.5/95%), residual bias and
   correlation, and calibration.
3. Preliminary `mean_model`/`pdstl` flights to check the model transfers.
   Then freeze the geometry, model and plans, and alternate the two
   conditions within each fan for the final comparison.
