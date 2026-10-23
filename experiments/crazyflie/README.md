# Crazyflie experiment

The drone flies a minimum-snap curve that is uploaded to it. pdSTL (or the
mean-model baseline) chooses the curve's **waypoints**. A degree-7
minimum-snap polynomial with fixed segment times goes through them. The
pieces are uploaded to the drone, and the onboard controller flies them
(`start_trajectory`).

The reference, `plans/reference.json`, is the **figure-8**: a vertical
lemniscate centred at (0.56, −0.98), 0.68 m wide and 1.5 m tall, at 0.30 m
altitude. It starts at the bottom tip, passes the goal at the top tip and
returns: 22 pieces of 1.0 s (22 s), max speed 0.241 m/s. The arena is in
`config.yml`: x [0, 1], y [−2, 0], with three measured boxes (obs_2,
obs_3, obs_4), all taller than the flight height. The figure-8 was sized
to clear them: about 2.5 cm beyond the 7 cm airframe radius at the
closest point (obs_2).

The task is: stay in the workspace, avoid obstacles, and reach the goal at
some step.

## How a flight works

1. **Positioning.** The Kalman filter is reset and the position estimate
   must settle. The drone can sit anywhere in the workspace; a settled
   position outside it (as without Lighthouse fixes) refuses takeoff.
2. **Upload, on the ground.** The onboard controller is set
   (`flight.controller`: 1 = PID, the default; 2 = Mellinger, which hovered
   10 cm low on our drone). The pieces are written to trajectory
   memory and defined, before arming. If the upload fails, nothing takes off.
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

drone position = curve, delayed + bias + a random error.

With the PID controller the drone trails the uploaded curve by a steady
delay (about 0.5 s on our drone). The delay is fitted per fan first; the
error is then measured behind the delayed curve, `e_k = p_k − r(t_k − d)`.

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

## Running the experiment

Run everything from this directory. For a new checkout, copy
`config.example.yml` to `config.yml`.

**Fan levels:** `off` (baseline), `low`, `medium`, `high`. Set the fan by
hand before each `fly`; `--fan` only labels the record.

| Phase | Fan levels | Flights | Purpose |
| --- | --- | --- | --- |
| Pilots | each level, before its data | 1–3 | check the drone and the airflow |
| Identification | off, low, medium, high | **12 completed per level** (48) | fit one model per level |
| Planning (offline) | low, medium, high | 2 plans per level | deterministic (`mean_model`) and `pdstl` |
| Evaluation | low, medium, high | **20 per condition per level** (120) | compare the two conditions |

`off` is the baseline: identification only, no plans and no evaluation.
The identification flights also show how the unadjusted figure-8 does
under each level.

### Before every session

1. `config.yml` has `measured: true`, and **every box inside the net is
   listed** under `obstacles` at its measured position. Boxes and fan stay
   where they are for the whole data collection.
2. Write down where the fan stands and where it points.
3. Lighthouse deck firmly seated, propellers undamaged, battery charged.
   Swap the battery when its **resting** voltage at the start of a flight
   is below 3.7 V.
4. In cfclient, hold the drone still: x/y steady to a few mm.

### 1. Pilot at each level (go / no-go)

```bash
python run.py fly --fan low --purpose pilot
python run.py analyze --trial latest
```

Start a level's identification flights only when its pilots show:

- the drone hovers steadily at the start before the curve (no bouncing of
  more than a few cm), and **does not fall** during the curve;
- cross-track error around 15–25 mm RMS, wobble under about 15 mm;
- every box at least about 10 cm from the drone's centre;
- `ctrltarget` vs `r(t)` under 1 cm (the alignment is right).

If the drone falls or bounces at a level, the airflow is too strong or
too turbulent there (for example in a box's wake): move or aim the fan,
or lower that level, before collecting data. A drone knocked out of the
air is not data the model can use.

### 2. Identification: 12 completed flights per level

```bash
python run.py fly --fan low --purpose identification
python run.py fit
```

Fly until the level has **12 flights that reached the figure-8**: attempts
that never started (no connection, start gate, positioning) do not count,
so just fly again. Run `fit` after each level and read its line: the
delay, the bias `b`, the stationary spread and the held-out coverage (near
39 / 86 / 95 %). Every third flight per level is held out. `fit` warns
below 8 fit flights per level; two is only the software minimum.

A flight ruined by the hardware (for example a Lighthouse-deck drop-out)
can be excluded by writing a reason into its `metadata.json` as
`"data_readiness"`.

### 3. Plan (offline)

Set `model_report` in `config.yml` to the newest report, then per level:

```bash
python run.py plan --condition mean_model --fan low --output plans/mean_low.json
python run.py plan --condition pdstl --fan low --output plans/pdstl_low.json
```

Check each plan: feasible, threshold met, and how far the waypoints moved
from the reference. How much the two plans differ is a **diagnostic, not
a gate**: similar plans can mean the uncertainty is too small to change
the decision, the task is easy, or the bound or the optimizer is
uninformative. Investigate which, and report it.

### 4. Evaluation: start only when

- all four levels have their 12 identification flights and `fit` shows
  coverage near nominal at every level;
- the six plans (low, medium, high × two conditions) are feasible and
  reviewed;
- the arena, boxes and fan placement are unchanged since identification.

Then fly **20 per condition per level**, alternating the two conditions
(deterministic, pdstl, deterministic, …), with the purpose `evaluation`:

```bash
python run.py fly --plan plans/mean_low.json --purpose evaluation --fan low
python run.py fly --plan plans/pdstl_low.json --purpose evaluation --fan low
```

### 5. Analyze

`python run.py analyze` writes `results/summary.md`: success rates with
Clopper–Pearson intervals, kept apart from the model scores. Pilots, dry
runs and unmeasured trials are never fitted, and evaluation data never
refits the model.

## Which flights the fit uses

Identification flights that completed are used, and so are flights that
aborted on a `TaskViolation` (up to the abort): dropping large deviations
would understate the noise. Data faults, errors and interruptions are
excluded.

**Falls** are reported, not fitted: a flight whose height dropped below
half the flight altitude during the curve (lost lift, a crash) is left out
of the error model, which is Gaussian and cannot describe it. `fit` lists
the falls per level and the fall rate (falls / flights that reached the
curve); report the fall rate as its own result.

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

## The IROS scenario

The first IROS draft flew a sine path past three boxes (irobot commit
`a2e4fc0`). `config.iros.example.yml` sets it up for this pipeline, with the same
model, planner and limits as the figure-8. `plans/iros_sine.json` is the
reference: the IROS waypoints, 9 pieces of 3.0 s (27 s), max speed
0.224 m/s, at 0.30 m altitude. The boxes keep their IROS sizes. obs_2 is
moved 0.255 m left, because the sine curve cut 3.9 cm into it at its
IROS position; the curve now clears the airframe by 2.6 cm, as the
figure-8 does.

```bash
cp config.iros.example.yml config.iros.yml  # place and measure the boxes
python run.py --config config.iros.yml fly --fan low --purpose identification
python run.py --config config.iros.yml fit
```

Pass `--config config.iros.yml` to every command. `fit` and `analyze` use
only the trials flown under the configured reference, so the two
scenarios never mix.

## Other commands

```bash
python run.py fly --fan off --dry-run                   # simulated drone
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
