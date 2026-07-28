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
estimate_tracking_covariance.py
                        offline: fits Sigma_k = Sigma0 + k*Q per fan level from
                        repeated figure8 deterministic flights; writes
                        calibrated_uncertainty.yml -- see
                        [Tracking-covariance calibration](#tracking-covariance-calibration)
components/
    config.py          thin config.yml loader + shared helpers (spline math,
                        waypoint validation/I/O) — edit config.yml, not this
                        file, to change input values
    planning_2d.py      2D-only: plain Environment/Planner/SingleIntegrator from
                        src/planning/ (x, y only) for the baseline mission, plus
                        the closed-form deterministic-path calculation
                        (nominal_safe_waypoints)
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
`obstacles`, `goal`, `start_xy`, `end_xy`, and `deterministic_path.via_points`.
Every stage reads geometry from here. Re-run step 2 and check the "Before"
plot panel — the deterministic path is a fixed curve through your via-points,
so it's on you to pick via-points that clear the obstacles; nothing computes
or checks clearance for you.

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
measured start is too far from `START_XY` (see
[Start-position calibration](#start-position-calibration)).

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

## Deterministic path

The 2D baseline's "deterministic" condition — and the pdSTL optimizer's warm
start — is `nominal_safe_waypoints()` (`components/planning_2d.py`): a sine
curve through `START_XY`, `config.yml`'s `deterministic_path.via_points`, and
`END_XY`, in that y-order. This curve **is** the deterministic path; it is
flown as-is, not fed into an optimizer.

y is linear in normalized progress `s ∈ [0, 1]`; x is the straight line
`START_XY → END_XY` plus one sine harmonic per via-point,
`x(s) = x_linear(s) + Σ a_k·sin(k·pi·s)`. Every harmonic vanishes at `s=0`
and `s=1`, so the coefficients `a_k` are solved by a single closed-form
linear system to hit each via-point exactly without perturbing start/end.

Via-points aren't checked against obstacle geometry — pick ones that clear
the arena (compare against the "Before" plot panel) the same way the gate
scenario's `nominal_gate_waypoints` via-points are chosen.

## Start-position calibration

An offline plan assumes an exact start position. Real flights drift, so at
flight time the drone hovers at the planned start, measures its real position
for ~2 s, and either aborts (offset too large) or shifts the whole plan by the
measured offset — a one-time pre-flight step.

There is no mid-flight replanning: the plan flies start-to-finish as given, at
the planned `U_MAX`, so actual flight matches the belief model's timing.

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

`rho_after` isn't tabulated here — it depends on the current `OBSTACLES`
geometry (which moves as the arena gets re-measured) and isn't cached
anywhere except the waypoints JSON files themselves; run `python run.py plan
--fan L` and read its printed `rho_after`, or `python -c` a quick read of
`waypoints/pdstl_fan<L>.json`. See
[Design notes](#design-notes--known-limitations) for why fan 6/12/16
generally fall short of `alpha=0.90`. `run.py fly --condition pdstl --fan L`
refuses to fly a plan with `rho_after<=0`; `--condition deterministic` is
unaffected at every fan level.

## Tracking-covariance calibration

The per-fan uncertainty above (`SIGMA0_PER_FAN`/`Q_STD`) is a placeholder:
one scalar per fan level, applied identically to every waypoint.
`estimate_tracking_covariance.py` checks whether that placeholder's
functional form — `Sigma_k = Sigma0 + k·Q` (an initial variance plus a
constant per-step increment) — actually matches real tracking error, using
the existing `figure8` scenario's deterministic flight and logging, unchanged:

```bash
# Repeat several times per fan level (10+ recommended), no new flags:
python run.py fly --scenario figure8 --condition deterministic --fan 2
python run.py fly --scenario figure8 --condition deterministic --fan 6
python run.py fly --scenario figure8 --condition deterministic --fan 12
python run.py fly --scenario figure8 --condition deterministic --fan 16

# Then, offline (no hardware):
python estimate_tracking_covariance.py            # all four fan levels
python estimate_tracking_covariance.py --fans 2 6  # or a subset
```

It reads the existing `components/logs/deterministic_figure8_fan<L>_run*_{commanded,actual}.csv`
files (nothing about logging changes), aligns each run's commanded waypoints
with the continuous actual-position trace (linear interpolation at each
waypoint's arrival time), computes the empirical tracking-error mean and
covariance **across runs** at each waypoint (not across the 10 Hz samples
within one run — the independent samples are the repeated flights), and fits
`Sigma0`/`Q` per fan level by least squares against that empirical
covariance. Writes `calibrated_uncertainty.yml` (fan → fitted `sigma0`,
`q_std`, and fit-quality metrics `r_squared`/`residual_rms`) and prints the
same per-fan report plus a `GOOD FIT`/`POOR FIT` verdict.

A poor fit (low `r_squared`, or a negative fitted parameter) is a stopping
point, not something to work around — it means the real tracking error
doesn't grow the way the current scalar model assumes, and the next step
would be deciding whether the planner needs a real per-waypoint covariance
schedule instead, not silently forcing this fit to look acceptable. This
script does not wire `calibrated_uncertainty.yml` into planning itself; it
only produces the calibration data for that future decision.

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
