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

`Planner.run_receding_horizon` is the only MPC loop. It optimizes a local window,
executes its first control, updates the state, shifts the remaining controls,
and repeats until the goal or step limit. `MPCResult` stores executed states,
applied controls, window plans, and the stopping reason.

## Scenarios and figures

- `configs/scenarios/altitude_safety.yaml`: altitude belief, atomic probability,
  controls, and final pdSTL interval.
- `configs/scenarios/reach_avoid.yaml`: one shared workspace, goal, and obstacle
  geometry for one-shot and MPC execution. Its `mpc` section holds step limit
  and seed. The default geometry has two blocks and one gap.
- `configs/scenarios/lane_change.yaml`: four-vehicle lane change with road
  safety, relative collision checks, and a timed target-lane dwell. This is the
  default live MPC run.
- `configs/scenarios/lane_merge.yaml`: selectable on-ramp merge where the ramp
  tapers into the main lane. It uses the same stochastic traffic and pdSTL
  safety machinery.

The canonical runners save a structured `.pt` result and a 300-dpi `.png` plus
vector `.pdf` figure in `outputs/`. Altitude and reach–avoid GIFs reveal the
predicted trajectory step by step. MPC and lane GIFs show every executed step,
its current plan, the lower satisfaction bound, and surrounding traffic when
present. Load trusted result files with
`torch.load(path, weights_only=False)`.

The publication plots show predicted belief means, selected 95% belief ellipses,
executed trajectories, predicate intervals, controls or optimization loss, and
per-window lower satisfaction bounds as appropriate. The ellipse uses the joint
95% radius for a two-dimensional Gaussian. The MPC and lane live view places the
environment at the center and shows sampled optimizer updates in graphs beside
it, then advances the scene after each executed step.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

Edit the `"run"` and `"skip"` flags beside each project block in `src/main.py`.
The current flags in that file select the projects to run. `show_plots=True`
displays each completed plot before saving it. `show_optimization=True`
reports sampled gradient descent iterations in the terminal. For MPC and lane
runs, the same live figure shows the candidate path and side graphs with an
interactive Matplotlib backend. Set `optimization_every=1` in `src/main.py` to
display every update; the default samples every 5 iterations. `live_plots=True`
shows the environment as planning and execution progress. The public
runners also accept `show`, `save`, `live_optimization`, and, for MPC and lane
runs, `live`. Their `max_steps` override is useful for bounded runs.

The deterministic baseline remains available for comparison of a stored lane
plan; it does not add another optimizer. Lane collision checks use configured
relative longitudinal and lateral bounds around the vehicle centers.

## Development checks

```bash
ruff format --check src tests
flake8 src
PYTHONPATH=src MPLBACKEND=Agg pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
