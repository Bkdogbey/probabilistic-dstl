# Crazyflie pdSTL experiment

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
| `config.yml`, `config.py` | All inputs, as one typed `Config` |
| `run.py` | Entry point: `plan`, `fly`, `fit`, `analyze` |
| `planner.py` | Task geometry and pdSTL spec, reference/excitation commands, both planners, plan files |
| `flight.py` | Preflight, start gate, 20 Hz velocity streaming, abort, landing, ROS Sugar launch, `SimDrone` |
| `logger.py` | `logs/<trial>/commands.csv`, `state.csv`, `metadata.json`; loading and time alignment |
| `tracking_model.py` | `TrackingModel` (first-order velocity lag with fan bias `c_f` and noise `Q_f`), fit, validation |
| `analysis.py` | Mission outcomes, success with Clopper–Pearson CIs, figures, single-trial check |

`planner.py` is not named `planning.py` because that would shadow the core
`planning` package in `src/`.

## Commands

Run from this directory. Real flights need ROS Sugar, cflib and irobot
(`conda activate ros_env && source ~/sugar_ws/install/setup.bash`,
`pip install cflib`, `pip install -e ~/Research/irobot`).

```bash
python run.py plan --condition deterministic                      # fixed reference
python run.py plan --condition deterministic --sequence steps_x   # identification excitation
python run.py fly --plan plans/<id>.json --purpose pilot --fan off --dry-run
python run.py fly --plan plans/<id>.json --purpose identification --fan off
python run.py analyze --trial <trial_id>                          # timing + commanded vs measured
python run.py fit                                                 # reports/model_<UTC>.json + .png
python run.py plan --condition pdstl --fan off                    # newest report by default
python run.py plan --condition mean_model --fan off
python run.py fly --plan plans/<id>.json --purpose evaluation --fan off
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
