# Crazyflie pdSTL experiment

Run the experiment with four commands: `plan`, `fly`, `fit`, and `analyze`.
Each flight streams world-frame horizontal velocities at a fixed altitude,
rotated into the body frame using measured yaw. ROS Sugar and irobot's
`CrazyflieBase` handle the hardware integration.

Use three fan conditions: **off, low, high**. Compare three planners:

| Condition | Reference or model | Uncertainty used in planning |
|---|---|---|
| `deterministic` | Fixed reference, shared across fan conditions | No |
| `mean_model` | Fitted mean dynamics | No |
| `pdstl` | Same fitted dynamics and task | Yes |

`mean_model` versus `pdstl` isolates the effect of uncertainty. They share
the initial control guess, velocity limits, control costs and optimization
budget.

## Current flight behavior

The current executor takes off, reaches the start, flies the mission, holds
briefly, and lands at the endpoint. The reference is still a polyline.
The requested smooth reference and automatic return to the start are
pending implementation; the commands below do not provide them yet.

Before a hardware pilot, fix `wait_for_positioning` so a missing or stale
initial state packet waits for fresh samples instead of applying `np.ptp`
to an empty window. The runbook adds no extra interactive preflight steps.

## 1. Set up the environment once

Use the existing ROS environment. Adjust the three local paths below if
your repositories or workspace are elsewhere.

```bash
conda activate ros_env
cd ~/Research/probabilistic-dstl/experiments/crazyflie
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e "../..[crazyflie]"
pip install -e ~/Research/irobot --no-deps
```

`--no-deps` skips irobot's unrelated trossen-arm dependency. ROS Sugar must
already be built in your ROS workspace. In every new terminal:

```bash
conda activate ros_env
source ~/sugar_ws/install/setup.bash
cd ~/Research/probabilistic-dstl/experiments/crazyflie
```

Planning, fitting and analysis use this environment too. Dry runs need
neither ROS nor a radio connection. Real flights check that ROS Sugar,
cflib and irobot are importable before connecting.

## 2. Enter the measured arena and fan settings

For a new setup only:

```bash
cp config.example.yml config.yml
```

If `config.yml` already contains your measurements, edit that file rather
than replacing it. It is ignored by git.

Enter the radio URI, workspace, obstacle footprints, airframe radius,
altitude, start, goal and reference waypoints. The template geometry is a
placeholder from the RA-L arena. Check that `flight.transit_z` clears the
measured obstacles if the drone must travel to the start above them.

Set up the three fan conditions in your local file:

```yaml
fans:
  "off": 0
  "low": null
  "high": null
```

Replace both `null` values with repeatable numeric settings chosen during
pilots. Record `fan_position: [x, y, z]` and
`fan_direction: [dx, dy]` using a unit direction vector. Keep the position
and direction fixed across conditions; change only the fan setting.
An airflow measurement at the flight area is useful if available.

**The program records the selected fan setting; it does not control the
fan. Set it manually before each flight.**

Once the geometry and fan setup are measured, set `measured: true`.
Identification and evaluation flights require this flag and a numeric
setting for the selected fan. Pilot flights can run before it is set.

## 3. Run the first deterministic pilot with the fan off

```bash
python run.py plan --condition deterministic
python run.py fly --purpose pilot --fan off --dry-run
python run.py fly --purpose pilot --fan off
python run.py analyze --trial latest
```

Inspect `plans/<plan id>.png` before the hardware flight. `fly` prints the
plan figure, duration, altitude, start and maximum speed, then asks once
whether the fan is set and the arena is clear.

`analyze --trial latest` prints command lateness, state-packet gaps, start
state, battery and task checks. Its commanded-versus-measured figure is
saved as `logs/<trial id>/flight.png`. Repeat a few pilots until execution
and logging are repeatable.

Then choose the active fan settings with pilot flights, one at a time:

```bash
python run.py fly --purpose pilot --fan low
python run.py analyze --trial latest
python run.py fly --purpose pilot --fan high
python run.py analyze --trial latest
```

For routine pilots, `fly` selects the newest matching plan and `latest`
selects the newest trial. Use explicit saved plan ids for final evaluation.
Pilots and simulated flights do not enter model fitting or evaluation.

## 4. Collect identification flights at each fan setting

Freeze the reference geometry, speed, start gate and fan setup before this
stage. Save the deterministic plan id printed in step 3.

Repeat the same reference **15 completed identification flights per fan**:
off, low and high. With `holdout_every: 3`, this gives 10 fitting and
5 held-out flights per fan. This is an example acquisition budget, not a
flight gate. The splitter counts completed identification flights only.

For each attempt, set the physical fan, run one flight, then inspect it:

```bash
python run.py fly --purpose identification --fan off --plan <reference-plan-id>
python run.py analyze --trial latest
```

Use the same two commands with `--fan low`, then `--fan high`. Replace
`<reference-plan-id>` with the actual id; do not type the angle brackets.
Run flights individually so you can reset the arena and inspect the data.
Every recorded attempt is retained, including aborted attempts.

For the current fitter, use reference flights from the same start. It
pools initial states per fan. The separate excitation sequences start at
`excitation.start`; mixing those starts into identification would distort
the initial belief used for reference planning. To inspect an excitation
without including it in fitting, use `--purpose pilot`:

```bash
python run.py plan --condition deterministic --sequence steps_x
python run.py fly --purpose pilot --sequence steps_x --fan off
```

The other available sequences are `steps_y` and `diagonal`. Their initial
belief must be handled separately before including them in model fitting.

## 5. Fit and inspect the held-out model checks

```bash
python run.py fit
```

This saves `reports/model_<UTC>.json` and its validation figure. Confirm
that the report includes off, low and high, each with held-out flights.
Inspect multi-step prediction error versus spread, residual bias and
correlation, and position coverage. A 2-D Gaussian predicts approximately
39.3%, 86.5% and 95% coverage for the reported ellipse thresholds.

The current code reports validation but does not enforce an acceptance
decision. If spread is underestimated or residuals have strong structure,
revise the model using fitting data and validate again before evaluation.
Do not tune on final evaluation flights. Save the chosen model report path.

The fitter currently reads all completed identification trials in `logs/`.
Keep one arena/fan protocol in that directory; archive older campaigns
outside it before starting a different protocol. Changing fan settings or
geometry requires new compatible data and a new model.

## 6. Generate both model-based plans for all three fans

Use the explicit report path so every plan uses the chosen model version:

```bash
python run.py plan --condition mean_model --fan off --model reports/<model-id>.json
python run.py plan --condition pdstl --fan off --model reports/<model-id>.json
python run.py plan --condition mean_model --fan low --model reports/<model-id>.json
python run.py plan --condition pdstl --fan low --model reports/<model-id>.json
python run.py plan --condition mean_model --fan high --model reports/<model-id>.json
python run.py plan --condition pdstl --fan high --model reports/<model-id>.json
```

Replace `<model-id>` with the printed model id. Inspect all six figures and
their feasibility and score outputs. A rejected plan is not flown; revise
the task or model transparently and regenerate plans before freezing them.

Run preliminary model-based flights as **pilots**, using their saved ids:

```bash
python run.py fly --condition mean_model --purpose pilot --fan off --plan <mean-plan-id>
python run.py fly --condition pdstl --purpose pilot --fan off --plan <pdstl-plan-id>
```

Repeat for low and high with the corresponding ids. Check that predictions
transfer to these commands. Then freeze the configuration, model report
and seven plan ids: one reference plus two model-based plans per fan.

## 7. Run the final evaluation

At each fan setting, compare all three conditions. Set the physical fan
before each attempt and use the frozen ids:

```bash
python run.py fly --condition deterministic --purpose evaluation --fan off --plan <reference-plan-id>
python run.py fly --condition mean_model --purpose evaluation --fan off --plan <mean-off-plan-id>
python run.py fly --condition pdstl --purpose evaluation --fan off --plan <pdstl-off-plan-id>
```

Repeat for low and high using their matching model-based plan ids. Rotate
the order between repetitions: deterministic/mean_model/pdstl, then
mean_model/pdstl/deterministic, then pdstl/deterministic/mean_model. Balance
fan-block order across sessions. Record unusual events with `--note`.

Choose the run count before evaluation. An example budget is:

| Flights | Per fan | Three fans |
|---|---:|---:|
| Identification: 10 fitting + 5 held out | 15 | 45 |
| Deterministic evaluation | 20 | 60 |
| Mean-model evaluation | 20 | 60 |
| pdSTL evaluation | 20 | 60 |
| Total, excluding pilots | 75 | 225 |

Start with pilots rather than immediately collecting this full dataset.
Do not refit or replace plans partway through final evaluation.

## 8. Analyze the evaluation flights

```bash
python run.py analyze
```

Outputs are `results/summary.csv`, `results/summary.md` and
`results/paths.png`. They compare condition × fan success counts and
Clopper–Pearson intervals, clearance, endpoint error and exclusions.

The current analysis groups by condition and fan, so one evaluation
campaign must use the frozen configuration and plans. Archive old campaign
logs before evaluating a changed setup; keep those records elsewhere.

The reported pdSTL score is a **pointwise stochastic-robustness score**,
not a bound on the probability of whole-flight success. Report it alongside
empirical success without treating the two as the same quantity.
Task-violation aborts count as failures. Data faults, errors and interruptions
are currently excluded with reasons; report those counts too. Every attempt
gets metadata, command and state files. Dry-run trials are excluded from
fitting and evaluation.

## Files and offline checks

| File | Role |
|---|---|
| `config.example.yml`, `config.py` | Configuration template and typed inputs |
| `run.py` | `plan`, `fly`, `fit`, `analyze` entry points |
| `planner.py` | Geometry, reference/excitation commands and both planners |
| `flight.py` | Flight phases, velocity streaming, aborts, ROS launch and simulator |
| `logger.py` | Per-attempt records and time alignment |
| `tracking_model.py` | Shared velocity lag; fan-specific bias, noise and initial belief |
| `analysis.py` | Task outcomes, timing checks, coverage and figures |

The planner file is named `planner.py` to avoid shadowing the core
`planning` package. The pdSTL core in `src/` is unchanged.

Run the experiment's offline checks from this directory:

```bash
python -m pytest tests -q
```

Abort handling hands control back to the high-level commander and attempts
landing. During a mission, fresh state/link, geofence, altitude and command
timing checks remain active. Each 0.2 s velocity is sent four times at
20 Hz; state logging carries `x y z vx vy yaw` in one 24-byte block.
