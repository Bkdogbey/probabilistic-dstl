# Probabilistic differentiable STL

pdSTL combines probability bounds for atomic events into Boolean and temporal
satisfaction intervals. A belief supplies `probability_bounds(event) -> [B, 2]`;
the pdSTL layer does not assume a particular uncertainty distribution. Gaussian
beliefs provide exact axis and half-space probabilities. Rectangle intervals use
Fréchet bounds. The lane example evaluates collision events on relative Gaussian
beliefs for three surrounding vehicles.

## Planning pipeline

```text
YAML scenario → dynamics and initial belief → rollout → atomic probability bounds
→ pdSTL formula → smooth lower score → bounded control optimization
→ final hard interval → PlanResult → publication figure
```

`Planner.optimize_window` optimizes only the smooth lower score, control effort,
and control smoothness. The hard interval is evaluated for reporting and optional
stopping. It is a pdSTL satisfaction interval, not an exact joint trajectory
probability. `PlanResult` holds controls, the predicted rollout, smooth score,
hard interval, one post-update loss per iteration, the final loss, and the beta
used for that result. Optional iteration observers receive a lightweight record
for the same post-update iterate. Saturated initial controls are moved inside
the tanh bound to preserve useful gradients.

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
- `configs/scenarios/reach_avoid.yaml`: a configurable, single-shot double-slit
  task. The belief stays inside the workspace, avoids every rectangular
  obstacle, and eventually reaches the goal. Bounds, names, styles, and the
  number of obstacles all come directly from YAML.
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

The double-slit publication plot contains only the configured environment, the
optimized belief mean, selected joint 95% Gaussian belief ellipses, start and
terminal markers, and the certified overall lower bound `P↓(φ)`. Lane plots
retain their execution diagnostics and live views. A second reach–avoid figure
shows the overall pdSTL interval as the prediction prefix grows over time; it
does not split the task into separate predicate probabilities. All planning
figures use the shared Matplotlib Tableau palette.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

Edit the `"run"` and `"skip"` flags beside each project block in `src/main.py`.
The current flags select only the one-shot reach–avoid example.
`show_plots=True` displays each completed plot before saving it. The
reach–avoid runner accepts `live=True` to update its candidate belief path and
overall pdSTL-over-time certificate during optimization. This remains a
single-shot optimization, not replanning. `optimization_every` controls the
live refresh cadence. Lane runners also accept `live`, `live_optimization`,
and `max_steps`.

Lane collision and road checks use the full axis-aligned vehicle footprint plus
configured safety margins. A merge succeeds only when the complete target-lane
dwell finishes inside the time window and before the ego front reaches the ramp
end.

Each planning window has a certified hard pdSTL interval. The combined plot
shows both endpoints; the purple smooth score is only the differentiable
optimization surrogate. Physical success describes the sampled execution, so
it is not itself a probability score. The final reported interval is the last
window solved before that execution terminated.

## Experiment notebooks

[`experiments/reach_avoid_demo.ipynb`](experiments/reach_avoid_demo.ipynb)
loads the editable double-slit configuration, displays the reach–avoid formula,
optimizes once through the production runner, and shows the final one-axis
environment figure, overall pdSTL-time figure, and animation. The notebook
contains no stored outputs.

[`experiments/lane_change_merge_demo.ipynb`](experiments/lane_change_merge_demo.ipynb)
is a from-dynamics-to-optimization walkthrough of both lane scenarios. It
visualizes the environments, derives the double-integrator belief rollout,
constructs the relative-traffic pdSTL task, optimizes one window step by step,
and then runs and visualizes both receding-horizon executions.

```bash
pip install -e ".[notebook]"
jupyter lab experiments/
```

Its figures and GIF are isolated under `outputs/experiments/reach_avoid/`.

## Development checks

```bash
pip install -e ".[dev]"
ruff format --check src tests
flake8 src
MPLBACKEND=Agg pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
