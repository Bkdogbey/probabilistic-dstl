# Probabilistic differentiable STL

pdSTL combines probability bounds for atomic events into Boolean and temporal
intervals. A belief supplies `probability_bounds(event) -> [B, 2]`; pdSTL does
not assume a distribution. Gaussian beliefs give exact axis and half-space
probabilities, Boolean operators use Fréchet bounds, and `Always` / `Eventually`
take the minimum / maximum over their window.

The result for a spec φ is the stochastic robustness interval
**[ρ̲_φ, ρ̄_φ]** (lower and upper). `beta=None` evaluates it exactly;
`beta > 0` gives a smooth outer interval (smooth ρ̲_φ ≤ exact ρ̲_φ) for
gradients.

## Code layout

| Piece | Role |
|---|---|
| `src/pdstl/` | the semantics: predicates, Boolean and temporal operators |
| `src/models/` | dynamics, Gaussian beliefs, belief rollouts, trajectory sampling |
| `src/planning/environment.py` | builds environments and their pdSTL specs (reach–avoid, lane) |
| `src/planning/planner.py` | the optimization: one window, or receding horizon |
| `src/planning/runners.py` | shared glue: set up, `solve`, `monte_carlo`, save |
| `src/experiments/reach_avoid.py`, `lane.py` | one file per experiment: `build(cfg)` and `run(config_path)` |
| `src/visualization/` | `figures`, `animation`, `live_plots` |
| `configs/scenarios/<experiment>/*.yaml` | the numbers for each case |
| `src/main.py` | runs the experiments; flip `"run"`/`"skip"` per block |

## Planning

`Planner.optimize_window` is direct shooting: from an initial control
sequence u⁽⁰⁾ it takes Adam steps on

    −w_φ · smooth ρ̲_φ(u) + w_u Σ‖u‖² + w_du Σ‖Δu‖²

and stops at the first iterate whose exact ρ̲_φ reaches the threshold
`alpha`, rather than pushing toward 1 (with `alpha: null`, or if α is never
reached, it runs all `max_iters` steps). It returns the iterate with the
highest exact ρ̲_φ, which under early stopping is the one that reached α. The initial guess's exact and
smooth ρ̲_φ are recorded (`initial_hard_lower`, `initial_smooth_lower`), so
each run prints what the optimization improved.

`Planner.run_receding_horizon` is the MPC loop used by the lane experiment:
plan a window, execute its first control, shift, and repeat until an outcome
(success, collision, road violation, deadline missed, planning failure, step
limit).

## Reach–avoid

Each file in `configs/scenarios/reach_avoid/` is one case of the same problem.
Add, move or remove regions:

```yaml
bounds: {x_range: [0, 10], y_range: [0, 10]}
obstacles:
  - {x_range: [3, 5], y_range: [4, 6]}
goal: {x_range: [7, 8], y_range: [8, 9], interval: [1, 60]}
visit_regions:
  - dwell: 5
    any_of:
      - {name: t1, x_range: [1, 2], y_range: [6, 7]}
      - {name: t2, x_range: [7, 8], y_range: [4.5, 5.5]}
```

The task is G[1,H](bounds ∧ ¬obstacles) ∧ F[goal interval](goal) ∧, for each
visit region, F[interval] G[0,dwell](region). `interval` (in steps) and `name`
are optional; `any_of` lists alternatives, any one of which suffices (the goal
accepts it too). In code: `Environment.set_bounds`, `add_obstacle`, `set_goal`,
`add_visit_region`.

`src/experiments/reach_avoid.py` builds the environment and spec, starts the
optimizer on the shortest collision-free route (`shortest_route`, configured
by `route: {clearance, resolution}`; the clearance is kept small, 0.05 m, so
the route only avoids starting on an obstacle and pdSTL adds the
uncertainty-aware separation), optimizes, and then checks the plan
by **Monte Carlo**: it samples noisy trajectories under the plan's controls and
scores each with the same spec (with zero variance every atom is 0 or 1, so the
operators reduce to Boolean logic). It prints ρ̲_φ next to the empirical P(φ)
with a 95% interval, and the figure shows both.

## Lane change and merge

`configs/scenarios/lane/lane_change.yaml` and `lane_merge.yaml`: traffic `x0`
(position, m) and `speed` (m/s), ego `x0_mean: [x, y, vx, vy]`. Collision and
road checks use the full vehicle footprint plus safety margins. A merge
succeeds only when the target-lane dwell finishes inside the time window and
before the ramp end. Each planning window reports its exact [ρ̲_φ, ρ̄_φ].

## Scope: what the numbers mean

1. **ρ̲_φ is a robustness, not P(φ).** `Always` takes the minimum over time of
   per-step bounds. That bounds each single step, not the whole trajectory, so
   ρ̲_φ can exceed the true probability that φ holds. On narrow passage,
   Monte Carlo gives P(φ) = 0.950, 95% interval
   [0.944, 0.956], against ρ̲_φ = 0.957. A sound probability
   bound over time (Fréchet: max(0, Σ lower − (n − 1))) exists but is much more
   conservative (0.55 there).
2. **ρ̄_φ is not a sound upper probability either:** `Eventually` takes the
   maximum over time for the upper value too. Read [ρ̲_φ, ρ̄_φ] as a
   robustness interval.
3. **Covariance does not depend on the controls** in the linear-Gaussian model,
   so the planner only moves the mean. Covariance steering is the natural
   extension.
4. **The start is outside the theory.** Far from the goal, ρ̲_φ is exactly 0
   (the Fréchet lower bound of a conjunction is 0 until its parts are likely),
   so a gradient method needs a starting path. The route only chooses where to
   start; ρ̲_φ and the Monte Carlo rate evaluate the plan that comes out.
5. **The lane first-window guess is a hand-shaped manoeuvre** (`warm_start:` in
   the lane YAMLs), the same kind of choice as item 4.
6. **Lane values are per window and open loop.** Success, collision and the
   other outcomes are sampled closed-loop executions, not guarantees.
7. **Reproducibility.** Reach–avoid is deterministic; lane and Monte Carlo are
   seeded from the YAML. Report numbers from CPU runs (`PDSTL_DEVICE=cpu`);
   CUDA can differ in the last digits.

## Other belief models

The planner, environment and formulas never see the distribution. A sample
(Monte Carlo) belief would add a `SampleBelief` whose bounds are the fraction
of particles in the event, widened by a finite-sample (e.g. Hoeffding) margin,
and a rollout that propagates particles with fixed noise draws and fills
`aux["mean_trace"]` and `aux["cov_trace"]` for the figures.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

Set each block in `src/main.py` to `"run"` or `"skip"`. Figures are shown,
then saved to `outputs/` (git-ignored) when closed: PNG, PDF, a GIF, and a
`.pt` result (`torch.load(path, weights_only=False)`).

## Development checks

```bash
pip install -e ".[dev]"
ruff format --check src tests
flake8 src
MPLBACKEND=Agg pytest -q -m "not slow"   # -m slow runs the full scenarios
```

## License

MIT. See [LICENSE](LICENSE).
