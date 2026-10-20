# Crazyflie experiment

The drone flies a minimum-snap curve that is uploaded to it. pdSTL (or the
mean-model baseline) chooses the curve's **waypoints**. A degree-7
minimum-snap polynomial with fixed segment times goes through them. The
pieces are uploaded to the drone, and the onboard controller flies them
(`start_trajectory`).

Two paths are frozen in `plans/`:

| Path | File | Arena | Shape |
| --- | --- | --- | --- |
| Figure-8 (the reference) | `reference.json` | `config.yml` (from `config.example.yml`): x [0, 1], y [−2, 0], no obstacles | Vertical lemniscate, centre (0.5, −1.0), 0.5 m wide, 1.5 m tall, 0.50 m altitude; from the bottom tip, past the goal at the top tip and back; 20 pieces of 1.0 s (20 s), max speed 0.245 m/s |
| Obstacle path | `obstacles.json` | `config.obstacles.yml`: three boxes | From (1.0, −2.0) between the boxes into the goal at (1.0, 0.0); 9 pieces of 2.0 s (18 s), 0.25 m altitude |

The task is: stay in the workspace, avoid obstacles, and reach the goal at
some step. Fly the obstacle path with its own arena:
`python run.py --config config.obstacles.yml fly --fan 0`.

## How a flight works

1. **Positioning.** The Kalman filter is reset and the position estimate
   must settle. The drone can sit anywhere in the workspace.
2. **Upload, on the ground.** The pieces are written to trajectory memory
   and defined, before arming. If the upload fails, nothing takes off.
3. **Takeoff** where it sits, a straight move to the start, then the
   **start gate** (averaged position, speed and height).
4. **The onboard controller flies the curve.** The host only monitors it:
   state freshness, the radio link, the geofence and a trailing 0.25 s
   altitude mean.
5. **Land in place.** Ctrl-C during the flight also lands.

The host time of the `start_trajectory` call is not the firmware's start
time. The drone also logs the setpoint it is tracking (`ctrltarget`), and
the analysis fits the playback start `t_play` by matching that setpoint to
the curve.

## The model, in plain words

drone position = curve + bias + a random error.

Under the model, the spread of that error settles to a fixed (stationary)
value instead of growing without limit. Per fan setting, on the 0.2 s grid:

```text
e[k+1] = R e[k] + (I - R) b + w[k],   R = diag(rho_x, rho_y),   w ~ N(0, Q)
```

This is what the model **assumes**. The held-out coverage checks show
whether real flights behave this way. `Q` is an *effective* covariance: it
also absorbs state-estimation error and model mismatch, so it is not pure
fan noise. The model also assumes that, at a given fan setting, the error
statistics are the same for every path, so a plan only moves the curve.
Only flights of other paths can test this.

The simulator generates data from this same model. Simulated flights
therefore test plumbing and parameter recovery only, never whether the
model suits the hardware.

## Pipeline

Run these from this directory. Copy `config.example.yml` to `config.yml`
for a new checkout.

1. **Measure the geometry.** Enter the workspace, any obstacles and the
   airframe radius in `config.yml`, then set `measured: true`.
2. **First hardware step:** one slow fan-off pilot flight.

   ```bash
   python run.py fly --fan 0 --purpose pilot
   python run.py analyze --trial latest
   ```

   Before any identification flights, check the alignment (the offset of
   `t_play` and whether it was identifiable), the `ctrltarget`-vs-`r(t)`
   residual (well under 1 cm), and the time-aligned and cross-track errors.
3. **Identification flights** of the reference at fan settings 0, 6, 12
   and 18.

   ```bash
   python run.py fly --purpose identification --fan 6
   ```

4. **Fit.**

   ```bash
   python run.py fit
   ```

   Every third flight per fan is held out. In the report (`reports/`),
   check:
   - coverage near 39 / 86 / 95 %;
   - `flights_by_step`: flights that aborted early drop out of later
     steps;
   - the residual lag-1 correlation (near 0 if the model fits);
   - the stationary spread per fan.

   Set `model_report` in `config.yml` to the report.
5. **Plan** both conditions per fan:

   ```bash
   python run.py plan --condition mean_model --fan 6 --output plans/mean_fan6.json
   python run.py plan --condition pdstl --fan 6 --output plans/pdstl_fan6.json
   ```

   Report how much the plans differ: waypoint distance, min clearance and
   exact scores. This is a **diagnostic, not a gate**. Similar plans can
   mean the uncertainty is too small to change the decision, the task is
   easy, or the bound or the optimizer is uninformative. Investigate which
   one it is, and report it.
6. **Evaluation flights**, in a balanced order with a fixed count:

   ```bash
   python run.py fly --purpose evaluation --fan 6
   python run.py fly --plan plans/mean_fan6.json --purpose evaluation --fan 6
   python run.py fly --plan plans/pdstl_fan6.json --purpose evaluation --fan 6
   ```

7. **Analyze.** `python run.py analyze` writes `results/summary.md`.
   Success rates (with Clopper–Pearson intervals) are kept apart from the
   model scores.

**Flight counts.** Two fit flights per fan is the software minimum, not
enough calibration. Plan for at least `model.min_flights_warning` (8) per
fan before trusting a fit.

Set the physical fan before running `fly`: **0** is off; **6, 12 and 18**
are the active settings. `--fan` labels the record; it does not operate the
fan. Pilots, dry runs and unmeasured trials are never fitted. Evaluation
data never refits the model.

## Which flights the fit uses

Identification flights that completed are used, and so are flights that
aborted on a `TaskViolation` (up to the abort): dropping large deviations
would understate the noise. Data faults, errors and interruptions are
excluded.

## Planning

The planner moves the waypoints `w_1..w_M`. The start `w_0` and the segment
durations are fixed (the reference's), and the reference waypoints are the
initial guess. Both conditions use the same Adam loop, iteration count,
learning-rate sweep and penalties. Only the task score differs:

| Condition | Smooth score (optimized) | Exact score (stored) | Threshold |
| --- | --- | --- | --- |
| `pdstl` | `spec.smooth_lower`, core geometric beta schedule | pdSTL lower bound | `planner.alpha` |
| `mean_model` | smooth STL robustness of the predicted mean | hard robustness | `mean_model.min_robustness` |

A probability and a robustness are on different scales, so the weights
are set as follows and recorded in each plan's `objective`:

- `w_lim` (speed and acceleration norm limits) is zero inside the limits,
  so it only acts on infeasible iterates.
- `w_snap` is set per condition, so that at the reference the snap term's
  gradient is `snap_fraction` (1%) of the task score's gradient.
- The learning rate is picked per condition from `lr_grid`, with
  `sweep_iters` iterations for each value. Both conditions get the same
  sweep budget.

Each iterate, and the reference itself, is rounded to the uploaded float32
rows, checked densely and scored exactly on those rows. The best feasible
candidate is kept. Ties go to the higher smooth score, then the lower
penalty, then the earlier iterate. A plan stores three separate results:

- `feasible` and its `problems`;
- `exact_score`;
- `threshold_met`.

`fly` refuses infeasible plans. A feasible plan that misses its threshold
flies only with `--allow-below-threshold`.

**Scope.** The pdSTL score is evaluated on the 0.2 s grid. Between grid
points, only the dense geometric check covers the path. That check samples
at 100 Hz plus every piece boundary, so it is not a proof of clearance
between samples.

## Other commands

```bash
python run.py fly --fan 0 --dry-run                     # simulated drone
python run.py import waypoints.csv --segment 2.0 --output plans/slow.json
python run.py export plans/reference.json --csv /tmp/reference.csv
```

`import` reads `x,y` rows; the first row is the start. It takes one
`--segment` duration or one value per segment with `--durations`, and it
never replaces an existing file. `export` writes the
[uav_trajectories](https://github.com/whoenig/uav_trajectories) CSV format.

## Files

| File | Purpose |
| --- | --- |
| `run.py` | Command line: import, export, fly, fit, plan and analyze |
| `minsnap.py` | Min-snap solve, evaluation, float32 rows and CSV |
| `trajectory.py` | Plan records: build, check, read, load and import |
| `flight.py` | Upload, takeoff, start gate, mission monitor, landing |
| `logger.py` | Trial records, onboard timestamps and playback alignment |
| `tracking_model.py` | Error model: fit and held-out validation |
| `planner.py` | Offline waypoint planning for both conditions |
| `geometry.py` | Workspace, obstacles, goal and sampled feasibility checks |
| `analysis.py` | Outcome summaries, plan previews and flight checks |
| `simulator.py`, `tests/` | Hardware-free drone and regression checks |

Each attempt saves `state.csv`, `target.csv` and `metadata.json` in
`logs/`. The metadata also holds the uploaded rows and the alignment
results. Mission scoring ends before landing.

## Installation and checks

One-time installation, from this directory: `pip install -e "../..[crazyflie]"`
and `pip install -e ~/Research/irobot --no-deps`. ROS Sugar must also be
available in the sourced ROS environment. Tests need neither cflib nor
hardware. From the repository root:

```bash
python -m pytest experiments/crazyflie/tests -q
```
