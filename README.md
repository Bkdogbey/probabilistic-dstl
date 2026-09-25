# Probabilistic differentiable STL

pdSTL combines probability bounds for atomic events into Boolean and temporal
satisfaction intervals. A belief supplies `probability_bounds(event) -> [B, 2]`;
the pdSTL layer does not assume a particular uncertainty distribution. Gaussian
beliefs provide exact axis and half-space probabilities. Rectangle intervals use
Fréchet bounds. The lane example evaluates collision events on relative Gaussian
beliefs for three surrounding vehicles.

`beta=None` evaluates exactly; `beta > 0` gives a smooth, **sound** outer
interval (smooth lower ≤ exact lower), so a smooth lower ≥ α certifies the exact
one.

## Planning pipeline

```text
YAML scenario → dynamics and initial belief → rollout → atomic probability bounds
→ pdSTL formula → sound smooth lower bound → bounded control optimization
→ highest exact lower bound → PlanResult → publication figure
```

`Planner.optimize_window` **maximizes the exact pdSTL lower bound**: it runs
`max_iters` Adam steps on the sound smooth bound (plus small `w_u`, `w_du`
regularizers, no shaping terms) and returns the iterate with the highest exact
lower bound. `alpha` only labels the result. `Planner.optimize_multistart` runs
one optimization per warm start and keeps the best certified plan. The exact
interval is a pdSTL satisfaction interval, not a joint trajectory probability.

`Planner.run_receding_horizon` is the generic MPC loop used by the lane
planners. It optimizes a local window,
executes its first control, updates the state, shifts the remaining controls,
and repeats until a terminal outcome. Lane outcomes are mutually exclusive:
success, collision, road violation, deadline missed, planning failure, or step
limit. `MPCResult` stores executed states, applied controls, window plans, and
the outcome.

## Scenarios and figures

- `configs/scenarios/altitude_safety.yaml`: altitude belief, atomic probability,
  controls, and final pdSTL interval.
- `configs/scenarios/narrow_passage.yaml`, `either_or.yaml`: reach–avoid
  examples (see below).
- `configs/scenarios/lane_change.yaml`: four-vehicle lane change with road
  safety, relative collision checks, and a timed target-lane dwell.
- `configs/scenarios/lane_merge.yaml`: selectable on-ramp merge where the ramp
  tapers into the main lane. It uses the same stochastic traffic and pdSTL
  safety machinery.

The canonical runners save a structured `.pt` result and publication figures
in `outputs/`. The reach–avoid runner saves one environment view as PNG and
PDF, plus a GIF that reveals the optimized trajectory and its current belief
ellipse. Lane runs produce one combined result and one local-scale trajectory
view. Their GIFs show every executed step, its current plan, the lower
satisfaction bound, and surrounding traffic. Load trusted result files with
`torch.load(path, weights_only=False)`.

Edit traffic directly in `configs/scenarios/lane_change.yaml` or
`configs/scenarios/lane_merge.yaml`: `x0` is the initial longitudinal position
in metres and `speed` is the longitudinal speed in m/s. The ego initial state is
`x0_mean: [x, y, vx, vy]` in the same file.

## Reach–avoid examples

One configurable planner covers every reach–avoid scenario. A config lists
boxes by role; the task is always the safe region, any goal, and, per `visit`
group, one target held for `dwell` steps:

```yaml
workspace: {x: [0, 10], y: [0, 10]}
obstacles: {block: {x: [3, 5], y: [4, 6]}}
goals: {goal: {x: [7, 8], y: [8, 9]}}          # reach any one
visit:                                          # enter one, stay `dwell` steps
  - dwell: 5
    regions: {t1: {x: [1, 2], y: [6, 7]}, t2: {x: [7, 8], y: [4.5, 5.5]}}
routes:                                         # warm starts, one run each
  via_t1: [[1.5, 6.5], [7.5, 8.5]]
  via_t2: [[7.5, 5.0], [7.5, 8.5]]
```

The robot is a double integrator, so plans are curved.
[`experiments/reach_avoid_examples.ipynb`](experiments/reach_avoid_examples.ipynb)
runs the stlpy NarrowPassage and EitherOr examples. A 95% ellipse may touch an
obstacle while a plan is certified: α = 0.9 allows about 1.3σ of clearance per
step, while a 95% ellipse has radius 2.45σ.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

Edit the `"run"` and `"skip"` flags beside each project block in `src/main.py`.
`show_plots=True` displays each completed plot before saving it; `live=True`
streams optimizer candidates. Lane runners also accept `live`,
`live_optimization`, and `max_steps`.

Lane collision and road checks use the full axis-aligned vehicle footprint plus
configured safety margins. A merge succeeds only when the complete target-lane
dwell finishes inside the time window and before the ego front reaches the ramp
end.

Each planning window has a certified exact pdSTL interval. The combined plot
shows both endpoints; the purple smooth score is the sound differentiable lower
bound the optimizer ascends. Physical success describes the sampled execution, so
it is not itself a probability score. The final reported interval is the last
window solved before that execution terminated.

## Experiment notebooks

[`experiments/reach_avoid_examples.ipynb`](experiments/reach_avoid_examples.ipynb)
runs the narrow-passage and either–or examples through `run_reach_avoid`.

[`experiments/lane_change_merge_demo.ipynb`](experiments/lane_change_merge_demo.ipynb)
is a from-dynamics-to-optimization walkthrough of both lane scenarios. It
visualizes the environments, derives the double-integrator belief rollout,
constructs the relative-traffic pdSTL task, optimizes one window step by step,
and then runs and visualizes both receding-horizon executions.

```bash
pip install -e ".[notebook]"
jupyter lab experiments/
```

## Development checks

```bash
pip install -e ".[dev]"
ruff format --check src tests
flake8 src
MPLBACKEND=Agg pytest -q -m "not slow"   # add -m slow for full scenario solves
```

## License

MIT. See [LICENSE](LICENSE).
