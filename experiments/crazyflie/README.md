# Crazyflie reach-avoid experiment

Real-world pdSTL experiment (paper Experiment 3). A Crazyflie flies from a start
to a goal through three obstacles under fan-induced disturbance, comparing a
**deterministic** nominal safe path against a **pdSTL**-optimised plan that
maximises the probabilistic satisfaction lower bound for a given fan level.

Two scenarios, planned by entirely separate code (neither file imports the
other — see [Scenarios](#scenarios)):
- **`baseline`** (default) — genuinely 2D. `components/planning_2d.py`.
- **`gate`** — genuinely 3D (climbs through a gate, descends, lands).
  `components/planning_3d.py`.

The pdSTL planner comes from this repo's `src/` (no vendored copies); the drone
hardware comes from the lab's `irobot` package.

## Layout

```
config.yml              all input values: arena geometry, per-fan uncertainty,
                        planner hyperparameters, flight params, gate geometry
run.py                 entry point: plan (offline) | fly (hardware) | analyze (offline)
waypoint_planning.py   shared CLI runner + plotting; dispatches to planning_2d/planning_3d
analyze_logs.py        post-flight: plot a logged run against its planned path
components/
    config.py          thin config.yml loader + shared helpers (spline math,
                        waypoint validation/I/O, geometry signature) — edit
                        config.yml, not this file, to change input values
    planning_2d.py      2D-only: plain Environment/Planner/SingleIntegrator from
                        src/planning/ (x, y only) for the baseline mission, plus
                        the closed-form deterministic-path sine-amplitude
                        calculation (_calculate_sine_amplitude)
    planning_3d.py      3D-only: Environment3D/Planner3D/SingleIntegrator3D
                        (Crazyflie-only (x,y,z) extension of src/planning/) +
                        gate-scenario construction (geometry values live in
                        config.yml, imported here)
    crazyflie.py        flight component (ros_sugar): calibrate, fly the plan
                        start->finish, return, land. Defines CrazyflieConfig.
    calibration.py      hover-and-measure start offset, abort-if-too-large, shift
    flight_logger.py    commanded/actual CSV logging, auto-incremented runs
    logs/               flight CSVs (gitignored)
waypoints/              generated pdstl[_<scenario>]_fan<L>.json (one per
                        scenario/fan-level pair) -- the deterministic path
                        itself is calculated on the fly, not cached to disk
plots/                  fan<L>[_<scenario>]_comparison.png,
                        <condition>_fan<XX>_run<NN>_actual.png
calibration/            generated empirical covariance profiles, one per fan
uncertainty_calibration.py
                        deterministic-log alignment + covariance estimation
```

## Quickstart

Fly **deterministic before pdstl** — it needs no generated plan, so it
exercises calibrate → fly → log on the simplest path first.

`config.yml`'s `trial:` section (`fan`, `condition`, `scenario`) is what
`plan`/`fly`/`analyze` act on when you don't pass `--fan`/`--condition`/
`--scenario` — edit it once and every subcommand below can be run with no
flags at all. The flags still exist and override `trial:` for one-off runs
(e.g. looping over fan levels without touching the file each time).

**1. Install (planning only, any machine)**
```bash
cd experiments/crazyflie
pip install torch numpy matplotlib pyyaml
```

**2. Smoke-test the planner (offline)**
```bash
python run.py plan --fan 2 --plot
```
Computes the 2D baseline's deterministic path (a closed-form sine-curve
calculation, not an optimizer — see [Deterministic path](#deterministic-path)),
then optimises the fan-2 pdSTL plan (single-shot from that deterministic
path's warm start), prints `rho_before`/`rho_after`, writes
`waypoints/pdstl_fan2.json`, saves `plots/fan2_comparison.png`, and opens it in
a window. If `import pdstl`/`planning` fails, run from inside
`experiments/crazyflie` (paths resolve relative to `components/config.py`).

**3. Measure the arena**

Edit `config.yml`'s `arena:` section: `flight_x_bounds`, `flight_y_bounds`,
`obstacles`, `goal`, `start_xy`, `end_xy`. Every stage reads geometry from
here. Re-run step 2 — it automatically recalculates the deterministic path's
amplitude (raising if no single-hump sine curve can clear every obstacle by
`deterministic_path.margin`), so there's no manual "eyeball the plot" step;
the "Before" plot panel is a sanity check, not the source of truth.

**4. Install the hardware stack (flight machine)**
```bash
pip install torch numpy cflib pyyaml
pip install -e <path-to-irobot-clone>   # e.g. pip install -e ~/Research/irobot
```
plus `ros_sugar` and ROS2. `torch` is required here too — `crazyflie.py`
imports `components/planning_2d.py` and `components/planning_3d.py` (for the
nominal paths), both of which always import `torch`/`pdstl`/`planning`.
`components/config.py` itself stays torch-free.

Default radio URI is `radio://0/80/2M/E7E7E7E780`. To fly a different drone,
pass a `hw_config` when building `CrazyflieConfig` in `run.py`'s `_fly()`:
```python
from irobot.src.robots.crazyflie.config import CrazyflieConfig as CrazyflieHwConfig
CrazyflieConfig(..., hw_config=CrazyflieHwConfig(uri='radio://0/.../...'))
```

**5. Fly deterministic**

Set `config.yml`'s `trial: {fan: 6, condition: deterministic}`, then:
```bash
python run.py fly
```
or override without editing the file: `python run.py fly --condition deterministic --fan 6`.
`--fan` only tags the logs — the path itself doesn't depend on fan level.
Watch for the calibration offset printout; it aborts before takeoff if the
measured start is too far from `START_XY` (see [Calibration](#calibration)).

**6. Plan and fly pdstl**
```bash
python run.py plan --fan 12 --plot     # once per fan level
python run.py fly --condition pdstl --fan 12
```
`fly --condition pdstl --fan L` flies `waypoints/pdstl_fan<L>.json`, which
records the fan it was generated for and is checked on load. Generate a fan's
plan once, fly it as many times as you like (set `trial.fan`/`trial.condition`
in `config.yml` and drop the flags once you're repeatedly working a single
fan). Repeat across fan levels (2, 6, 12, 16) to build out the comparison —
`--fan` overrides `trial.fan` per invocation, which is the easiest way to
batch this in a shell loop without editing the file each time. `run.py fly`
refuses to fly a plan that never converged (`rho_after<=0`) with a clear
error — see [Per-fan uncertainty](#per-fan-uncertainty) for which fans
currently work. For the gate-flythrough mission instead of the baseline
reach-avoid, add `--scenario gate` (or set `trial.scenario: gate`) — see
[Scenarios](#scenarios).

**7. Analyze**
```bash
python run.py analyze                                   # trial.condition/trial.fan, latest run
python run.py analyze --condition pdstl --fan 12 --run 3
python run.py analyze --all                             # every condition/fan pair
python run.py analyze --summary                         # cross-run table + rollup, no plots
```
Writes `plots/<condition>_fan<XX>_run<NN>_actual.png`: actual flown path,
commanded waypoints, and planned path on one arena drawing; unsafe samples
(inside an obstacle) marked red. `--summary` instead prints every logged run
(not just the latest per cell) plus a per-`(condition, fan)` rollup — run
count, crash rate, mean unsafe fraction, mean duration — for judging progress
across a batch of flights.

## Logging

`fly` writes `components/logs/<condition>_fan<XX>_run<NN>_<ts>_{commanded,actual}.csv`
— one row per commanded waypoint, and 10 Hz sampled real position. Run number
auto-increments per `(condition, fan)`. The return-to-start leg after each
trial flies but isn't logged.

Every row carries a `safe` flag (0/1) plus per-obstacle `outside_obsN` flags —
an obstacle's boundary counts as inside it (touching it is unsafe, not just
overlapping it). Two independent filename tags summarize a trial at a glance:

- `_CRASH` — the trial raised mid-mission (obstacle collision, tracking loss,
  hardware fault, ... any failure to reach the currently-commanded waypoint),
  including a calibration abort.
- `_VIOLATION` — at least one actual sample had `safe=0`. Independent of
  `_CRASH`: a trial can clip an obstacle boundary and recover (violation, no
  crash) or fail for an unrelated reason without ever entering an obstacle
  (crash, no violation). Both tags can appear on the same run.

A trial that never moves more than `START_TOLERANCE` from its own start
position (e.g. a calibration abort that fires before any real flight) isn't
saved at all — no files are written and the run number isn't consumed, so it
doesn't count toward the run quota for that `(condition, fan)` cell.

## Per-fan uncertainty

Following the paper's Crazyflie experiment, the fan level selects an
empirically characterized initial per-axis variance `SIGMA0_PER_FAN`.
Propagation uses the shared `Q_STD = 0.01 m/step`, so belief grows as
`Σ_t(fan) = SIGMA0_PER_FAN[fan] + t·Q_STD²`:

| Fan | Σ0 (m²) | q_std (m/step) | variance after 10 steps (m²) |
|----:|--------:|---------------:|--------------------------------:|
| 2   | 0.001 | 0.010 | 0.002 |
| 6   | 0.006 | 0.010 | 0.007 |
| 12  | 0.020 | 0.010 | 0.021 |
| 16  | 0.050 | 0.010 | 0.051 |

Fans 2/6/12 correspond to paper Settings 1–3. Fan 16 is an explicitly
uncalibrated extrapolation. The final column uses the baseline horizon
(`T=10`); the gate extension accumulates six additional propagation updates.

### Automatic calibration from flight logs

After collecting repeated deterministic baseline flights for one fan, build
an empirical full-covariance profile with:

```bash
python run.py calibrate-uncertainty --fan 2 --min-runs 10
```

The command ignores crashed trials, interpolates the actual Lighthouse trace
at each commanded waypoint-arrival timestamp, resamples every run to the
planner's `T+1` steps, and writes `calibration/uncertainty_fan<L>.json`.
The baseline planner automatically uses that profile when present; without a
profile it falls back to `SIGMA0_PER_FAN + t·Q_STD²`. Re-running calibration
changes its generation identifier, so flight-time validation rejects waypoint
plans made with the previous profile until they are regenerated.

Each new flight log records `BASELINE_PATH_ID`, which is now derived
automatically from a signature over the obstacles/start/end/margin that
determine the calculated deterministic path (see
[Deterministic path](#deterministic-path)) rather than hand-edited.
Calibration accepts only runs whose path identifier matches the current 2D
baseline, preventing residuals from a since-changed deterministic path from
silently configuring the current planner. After any change that shifts the
signature (obstacle/start/end geometry or `deterministic_path.margin`),
collect a fresh batch of deterministic runs before recalibrating -- any
`calibration/uncertainty_fan<L>.json` generated under an older signature is
automatically ignored (not deleted) until then, including the
`baseline_path_id: "pchip_legacy"` fan-2 calibration file already in this
repo, which predates this refactor and stays inert until fan 2 is
recalibrated.

The calibration also records mean tracking residual separately from covariance.
The current paper-faithful planner keeps the nominal state as the belief mean
and uses only the measured covariance profile. Inspect a large nonzero mean
residual as evidence of systematic tracking or frame-offset bias rather than
silently treating it as random noise.

`rho_after` isn't tabulated here — it depends on the current `OBSTACLES`
geometry (which moves as the arena gets re-measured) and isn't cached
anywhere except the waypoints JSON files themselves; run `python run.py plan
--fan L` and read its printed `rho_after`, or `python -c` a quick read of
`waypoints/pdstl_fan<L>.json`. See
[Design notes](#design-notes--known-limitations) for why fan 6/12/16
generally fall short of `alpha=0.90`. `run.py fly --condition pdstl --fan L`
refuses to fly a plan with `rho_after<=0`; `--condition deterministic` is
unaffected at every fan level.

## Calibration

An offline plan assumes an exact start position. Real flights drift, so at
flight time the drone hovers at the planned start, measures its real position
for ~2 s, and either aborts (offset too large) or shifts the whole plan by the
measured offset — a one-time pre-flight step.

There is no mid-flight replanning: the plan flies start-to-finish as given, at
the planned `U_MAX`, so actual flight matches the belief model's timing.

## Deterministic path

The 2D baseline's "deterministic" condition — and the pdSTL optimizer's warm
start — is `nominal_safe_waypoints()` (`components/planning_2d.py`): a single
left/right-bending sine curve, `x(s) = x_linear(s) - A*sin(pi*s)` for
normalized path progress `s` in `[0, 1]`. This curve **is** the deterministic
path; it is flown as-is, not fed into an optimizer.

`A` is calculated in closed form, not hand-tuned and not gradient-descent
optimized (`_calculate_sine_amplitude`, `components/planning_2d.py`): each
obstacle can be passed on its left (curve stays below `x_min`) or right
(stays above `x_max`), and for a fixed side the amplitude needed to clear it
by `deterministic_path.margin` is a closed-form expression evaluated over the
obstacle's y-projected range. With 3 obstacles there are only 8 possible
left/right assignments, so the calculation just enumerates them, discards any
assignment whose per-obstacle bounds don't intersect into a feasible
amplitude interval, and keeps the feasible one with the largest real
worst-case clearance (checked with the actual box-distance, not just the
linear bound). Raises `ValueError` if no assignment is feasible at all — this
single-hump curve family can't thread every obstacle layout; a tighter one
would need hand-picked via-points instead, like the gate scenario's
`nominal_gate_waypoints`.

`BASELINE_PATH_ID` (used to tag flight logs and calibration files) is a
sha256 signature (`components/config.py`'s `geometry_signature_2d()`) over
exactly the inputs the calculation depends on — obstacles, start/end points,
and `deterministic_path.margin` — so it changes automatically whenever the
calculated path would actually change, and stays fixed across edits (e.g.
planner hyperparameters) that don't affect it.

## Scenarios

Every `plan`/`fly` command takes `--scenario {baseline,gate}` (default
`baseline`):

- **`baseline`** — 2D reach-avoid. `components/planning_2d.py` builds a plain
  `Environment`/`Planner`/`SingleIntegrator` from `src/planning/`; optimizer
  state is strictly (x, y). Every waypoint flies at the fixed `Z_HEIGHT`,
  applied as a constant after optimizing. `--plot` draws flat 2D axes
  (`waypoint_planning.py`'s `_plot_2d`/`_draw_env_2d`).
- **`gate`** — 3D. `components/planning_3d.py` builds
  `Environment3D`/`Planner3D`/`SingleIntegrator3D`. Climbs to fly through a
  gate mounted on a pole at `(0.5, -1.25)`, then descends to `POST_GATE_Z`
  (0.3 m, a hard altitude ceiling — `POST_GATE_Z_BAND`) to avoid the obstacles
  at low altitude before landing. Gate geometry values (`GATE_X`/`GATE_Y`/
  `GATE_Z`/`GATE_T`/...) live in `config.yml`'s `gate:` section; the
  construction that builds an `Environment3D` from them lives exclusively in
  `planning_3d.py` — see [Design notes](#design-notes--known-limitations) for
  the gate's own convergence limits.

```bash
python run.py plan --fan 2 --scenario gate --plot   # optimise + save + plot
python run.py fly  --fan 2 --scenario gate           # fly it (needs hardware)
```

`--scenario gate` writes/reads `waypoints/pdstl_gate_fan<L>.json` (distinct
from the baseline's `pdstl_fan<L>.json`), so both scenarios' plans coexist per
fan level. Landing needs no extra code — the existing post-trial
`_return_to_start()` leg in `crazyflie.py` already descends to `LAND_Z` after
any mission, gate or baseline.

`config.yml` holds every input value used by both scenarios (arena bounds,
obstacle/goal geometry, per-fan uncertainty, planner hyperparameters, flight
params, and the gate-only geometry) — `components/config.py` just loads it;
no scenario-specific construction happens there. `waypoint_planning.py`'s
`run_plan()` is a one-line dispatch to `_run_plan_2d`/`_run_plan_3d`, each
calling into `planning_2d`/`planning_3d` respectively.

## Optimiser knobs

All in `config.yml`'s `planner.config` section (loaded into `PLANNER_CONFIG`
by `config.py`): paper-faithful (`w_phi`, `w_u`=λ, `alpha`), heuristic shaping
(`w_dist`, `w_obs`, `w_visit`, `obs_margin` — speeds convergence, not in the
paper's objective, set to 0 for the clean objective), and numerical (`lr`,
`max_iters`, `min_iters`, `converge_patience`, `loss_tol`). `scale` is
documented as controlling STL min/max smoothing but is currently inert — see
[Design notes](#design-notes--known-limitations).

`run_plan()` optimises single-shot: one `planner._optimize_window()` call
seeded from the deterministic path's warm start. (A multi-start variant —
several randomly perturbed warm starts, keep the exact-best — was tried and
found real gains at fan 6/12, but was removed for simplicity and planning
speed; see git history if you want to revive it.)

## Troubleshooting

- **matplotlib `CXXABI_... not found`** on `--plot`: conda libstdc++ mismatch.
  `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" python run.py plan --fan 2 --plot`.
- **`No optimised waypoints for fan L`**: generate it first with
  `python run.py plan --fan L`.

## Design notes & known limitations

Background for the numbers above — not needed to fly, useful if you're
tuning the optimiser or investigating why a fan level won't converge.

**Why fan 6/12/16 historically don't reach `alpha=0.90`:** the STL spec ANDs
~5 constraints (avoid each of 3 obstacles, reach goal, stay in bounds) via
the sound but conservative Frechet lower bound, `P(A₁∧...∧Aₙ) ≈ max(0, ΣPᵢ −
(n−1))` — to hit 0.90 combined with `n=5`, every individual constraint needs
~0.98 probability, which gets structurally harder as position uncertainty
grows faster with fan speed and as obstacles sit closer together (less room
for the belief's uncertainty ellipse to clear more than one constraint at
once). `obs_2`/`obs_3` were repositioned specifically to attack this (see
`arena.obstacles` in `config.yml`) — re-plan and check the new `rho_after`
rather than assuming the old numbers. A multi-start search (try several perturbed
optimizer inits, keep the best) measurably narrowed the gap before being
removed for simplicity — re-tuning `PLANNER_CONFIG` weights alone is a
weaker lever than either of those two, since the Frechet bound itself is
what's limiting, not the search quality.

**The `scale` (STL-smoothing) knob is dead:** `src/planning/planner.py`'s
`_optimize_window` calls `phi(traj)` with no `scale` kwarg, so every STL
operator's `scale=-1` default (exact, unsmoothed) is always used regardless
of `PLANNER_CONFIG['scale']`. Confirmed by testing: manually forcing
smoothing on doesn't reliably help (helped a little at one setting, hurt at
others) and doesn't move the fan 12/16 numbers meaningfully. Not fixed here
since it's a pre-existing gap in the shared library (affects every scenario
config under `configs/scenarios/`, not just this one) — worth a real
investigation on its own, not a blanket patch.

**Gate scenario currently only reaches a good plan at fan 2** (`rho_after ≈
0.89`). At fan 6/12/16, the physical gate's narrow ~8×6.75in opening combined
with those fans' larger position uncertainty crashes the combined
satisfaction probability to exactly 0 — the same Frechet-bound conservatism
as above, compounded by the gate scenario's extra chained constraints (timed
gate visit + post-gate altitude ceiling on top of the baseline's terms). If
you need higher-fan gate plans, start by loosening `gate.y_margin` in
`config.yml` (an unmeasured engineering default, not a physical constraint).
