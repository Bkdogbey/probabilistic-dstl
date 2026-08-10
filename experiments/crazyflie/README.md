# Crazyflie experiment

The experiment supports a 2D baseline path and a smooth 3D figure-eight. Both
use the same dimension-aware planner. The deterministic figure-eight is the
100-point Gerono curve configured in `components/config.yml`; it starts at
0.20 m, passes through the centre crossing at 0.275 m, rises smoothly to
0.60 m at the top, and flies at 0.30 m/s.

## Run

Choose the run at the top of `src/main.py`:

```python
CRAZYFLIE_ACTION = 'plan'       # plan | fly | analyze
CRAZYFLIE_FAN = 12              # 2 | 6 | 12 | 16
CRAZYFLIE_CONDITION = 'pdstl'   # pdstl | deterministic
CRAZYFLIE_SCENARIO = 'figure8'  # baseline | figure8
CRAZYFLIE_PLOT = False
CRAZYFLIE_ANALYSIS = 'latest'   # latest | all | summary
CRAZYFLIE_RUN_NUMBER = None
```

Then run:

```bash
python src/main.py
```

- `plan` optimizes and signs the selected pdSTL plan.
- `fly` runs the selected deterministic or pdSTL path.
- `analyze` plots or summarizes recorded flights.

`components/config.yml` contains geometry, dynamics, hardware, safety, and approved
uncertainty values. Changing a planning input invalidates existing plans.

## Safety

Before arming, flight requires a stable position estimate (settled spread and
enough visible base stations per `min_base_stations`). When placed within
`start_tolerance` of the mission start, the drone takes off directly to the
planned mission altitude. If it is farther away, it uses `return_z` (above
every configured obstacle) as a safe transit altitude. pdSTL flight
additionally requires a current signed plan with `rho_after >= planner.alpha`.

The nominal figure-eight is a centreline, not an airframe-clearance guarantee.
Only fly the deterministic path after checking the physical arena. The pdSTL
condition may move away from the nominal curve to satisfy its obstacle constraints.

After the trial, the drone retraces the flyable mission waypoints instead of
climbing to `return_z` (or skips the return when it already finished at the
start). At the start position it descends at
`landing_velocity` to `land_z` (0.1 m), waits for the position to settle, and
only then issues the final landing command.

Only the forward experiment is logged. Takeoff, transit-to-start, return, and
landing are outside the trial. Every exit path stops logging, lands when
airborne, disarms, and disconnects.

## Data

```text
logs/2d/             baseline flight logs
logs/3d/             figure-eight flight logs
waypoints/           signed pdSTL plans
plots/               current planning, flight, and calibration evidence
calibration/reports/ reviewed covariance reports
```

Only three figure families are retained:

- `planning_<scenario>_fan<level>.{png,pdf}` shows nominal and pdSTL paths,
  obstacles, satisfaction probability, and sparse one-standard-deviation bars.
- `flight_<condition>_<scenario>_fan<level>_run<run>.{png,pdf}` compares a
  flight with the commands recorded in that same run and marks unsafe samples.
- `calibration_<report>.{png,pdf}` compares raw mean tracking error with the
  response-model residual and shows residual variance, its stationary estimate,
  and the conservative upper bound.

Parameter-sweep, margin-tuning, and other development plots are not retained in
the experiment output directory.

## Offline covariance estimation

New figure-eight CSV rows contain two provenance fields:

- `campaign` is `pilot` or `final`, selected by `calibration.active_campaign`.
- `profile_signature` hashes the path, cruise speed, sample count, estimator,
  start-position settings, workspace, and obstacle layout.

Before mission logging begins, execution waits until the measured 3D position
is within 0.03 m of the commanded start, lets it settle for one second, and
checks once more. If this does not happen within 15 seconds, the preflight is
aborted without creating a calibration run.

The estimator rejects crashed, violating, incomplete, implausible, stale-profile,
and wrong-campaign attempts with an explicit reason. Actual positions are
interpolated at commanded arrival timestamps before computing tracking error.

The planner-facing mean is response-aware. On the enabled horizontal axes it
uses the linear first-order controller model

```text
dv/dt = (u - v) / tau
dp/dt = v
```

The exact zero-order-hold solution is used at each planner step. The current
pilot does not support the same lag model on z, so vertical mean response is
instantaneous and its remaining structure stays in the residual diagnostics.
The position belief then uses

```text
P_0 = 0
P_f,k = diag(r_f,x, r_f,y, r_f,z),  k >= 1
```

where `r_f` is the fan-conditioned stationary variance left after response
prediction. A constant fan-conditioned residual mean is reported separately.
Because that one mean cannot represent any remaining waypoint-dependent bias,
`r_f` is computed around the same pooled mean consumed by the planner; it
therefore conservatively includes unresolved phase structure. Waypoint-centered
covariance is retained only as a repeatability diagnostic. The estimator also
retains raw tracking error and bias-inclusive residual MSE so deterministic lag
cannot be quietly discarded or mislabeled.

The response time constants are shared across fans and the residual statistics
are fan-specific. Bootstrap samples resample complete flights and refit the
response before estimating residual covariance, preserving temporal dependence
and response-parameter uncertainty. No random-walk growth or arbitrary
covariance value is used.

Response fit quality is checked with tracking-error R-squared on every enabled
axis. Residual stationarity is checked on ten-waypoint variance bins; the
largest binned variance may be at most twice its pooled variance. Either failure
stops final acceptance instead of forcing the model.

### Pilot campaign

The repository is currently configured for a disposable five-run fan-2 pilot:

```yaml
calibration:
  active_campaign: pilot
```

Run five complete deterministic figure-eight flights with fan 2, then generate
the diagnostic report:

```bash
PYTHONPATH=src python3 experiments/crazyflie/estimate_covariance.py --mode pilot
```

Pilot reports are named `pilot_covariance_<timestamp>.yml`, always carry
`accepted: false`, and never expose approved configuration values. Stop and fix
the flight/logging setup if any of the five attempts is excluded.

### Final campaign

After measuring and entering the final obstacle coordinates, remove the pilot
flight files and pilot report, then deliberately change:

```yaml
calibration:
  active_campaign: final
```

Changing obstacles changes the profile signature and invalidates existing
figure-eight plans. Collect 20 valid runs per fan, interleaving fan levels in
blocks (`2, 6, 12, 16`, repeated) to reduce battery and time drift. Failed
attempts remain on disk and appear in the report with exclusion reasons; keep
collecting until every fan has 20 valid runs.

Generate the final report with:

```bash
PYTHONPATH=src python3 experiments/crazyflie/estimate_covariance.py --mode final
```

Final acceptance requires all 80 valid runs, complete accounting for every
excluded attempt, an acceptable response fit, finite nonnegative per-axis
residual variances, acceptable residual stationarity, and at least 95% valid
bootstrap samples. A failed final report is still written for diagnosis, exits
nonzero, and contains no `approved_values`. An accepted report records the
fitted response time constants, uses the upper 95% bootstrap bound for each
fan's XYZ stationary residual variance, and records the separately estimated
XYZ residual mean.

Review an accepted report, copy only its `approved_values` into `uncertainty`,
set `uncertainty.source_report` to that report, and regenerate the four
figure-eight pdSTL plans. The estimator never edits `config.yml` itself.

## Tests

```bash
python -m pytest experiments/crazyflie/tests -q
```

Hardware acceptance remains a manual deterministic-flight check.
