# Crazyflie experiment

The experiment supports a 2D baseline path and a smooth 3D figure-eight. Both
use the same dimension-aware planner. The deterministic figure-eight is the
100-point Gerono curve configured in `components/config.yml`; it starts at
0.20 m, passes through the centre crossing at 0.275 m, rises smoothly to
0.60 m at the top, and flies at 0.30 m/s.

## Run

Choose the run at the top of `src/main.py`:

```python
ACTION = 'plan'            # plan | fly | analyze
FAN = 12                   # 2 | 6 | 12 | 16
CONDITION = 'pdstl'        # pdstl | deterministic
SCENARIO = 'baseline'      # baseline | figure8
PLOT = False
ANALYSIS = 'latest'        # latest | all | summary
RUN_NUMBER = None
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
additionally requires a current signed plan with positive `rho_after`.

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
plots/               planning and analysis figures
calibration/reports/ reviewed covariance reports
```

## Offline covariance estimation

New figure-eight CSV rows contain two provenance fields:

- `campaign` is `pilot` or `final`, selected by `calibration.active_campaign`.
- `profile_signature` hashes the path, cruise speed, sample count, estimator,
  workspace, and obstacle layout.

The estimator rejects crashed, violating, incomplete, implausible, stale-profile,
and wrong-campaign attempts with an explicit reason. It reports centered
covariance and mean tracking error separately, then fits the planner with the
bias-inclusive model
`E[||error_k||²]/3 = sigma0_fan + k*q_var`. The intercept is nonnegative and
fan-specific; `q_var` is nonnegative and shared across fans, matching
`config.yml`.

Every report also carries a `pooled_model` section: a constant per-fan
`E[||error||²]/3`, pooled across all waypoints with its own bootstrap CI, and
no `k`-dependence. This is the model actually validated in the pdSTL paper's
real-world section (arXiv:2606.19561, Sec. III.C) — `Sigma_0` pre-characterized
from pooled tracking-error residuals per fan. It is diagnostic only: it never
gates `accepted` and never feeds `approved_values`. Use it as a fallback
reading when `joint_fit.r_squared` is poor, which happens when tracking error
is driven by trajectory curvature rather than growing linearly with waypoint
index.

### Pilot campaign

The repository is currently configured for a disposable five-run fan-2 pilot:

```yaml
calibration:
  active_campaign: pilot
```

Run five complete deterministic figure-eight flights with fan 2, then generate
the diagnostic report:

```bash
python experiments/crazyflie/estimate_covariance.py --mode pilot
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
python experiments/crazyflie/estimate_covariance.py --mode final
```

Final acceptance requires all 80 valid runs, a nonnegative finite fit,
`R² >= 0.70` on ten-waypoint variance bins against the fan-specific constant
baseline, and at least 95% valid bootstrap samples. Binning is used only for
fit diagnosis; parameters are fitted from all 100 waypoints. A failed final
report is still written for diagnosis, exits nonzero, and contains no
`approved_values`. An accepted report uses the upper 95% bootstrap bounds for
conservative `sigma0_per_fan` and `q_std` values.

Review an accepted report, copy only its `approved_values` into `uncertainty`,
set `uncertainty.source_report` to that report, and regenerate the four
figure-eight pdSTL plans. The estimator never edits `config.yml` itself.

## Tests

```bash
python -m pytest experiments/crazyflie/tests -q
```

Hardware acceptance remains a manual deterministic-flight check.
