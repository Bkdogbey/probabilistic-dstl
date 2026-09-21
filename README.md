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
and repeats until a terminal outcome. Lane outcomes are mutually exclusive:
success, collision, road violation, deadline missed, planning failure, or step
limit. `MPCResult` stores executed states, applied controls, window plans, and
the outcome.

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

The canonical runners save a structured `.pt` result and publication figures
in `outputs/`. Lane runs produce one combined result and one local-scale
trajectory view as both 300-dpi PNG and vector PDF. Altitude and reach–avoid
GIFs reveal the predicted trajectory step by step. MPC and lane GIFs show every
executed step, its current plan, the lower satisfaction bound, and surrounding
traffic when present. Load trusted result files with
`torch.load(path, weights_only=False)`.

Edit traffic directly in `configs/scenarios/lane_change.yaml` or
`configs/scenarios/lane_merge.yaml`: `x0` is the initial longitudinal position
in metres and `speed` is the longitudinal speed in m/s. The ego initial state is
`x0_mean: [x, y, vx, vy]` in the same file.

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

Lane collision and road checks use the full axis-aligned vehicle footprint plus
configured safety margins. A merge succeeds only when the complete target-lane
dwell finishes inside the time window and before the ego front reaches the ramp
end.

Each planning window has a certified hard pdSTL interval. The combined plot
shows both endpoints; the purple smooth score is only the differentiable
optimization surrogate. Physical success describes the sampled execution, so
it is not itself a probability score. The final reported interval is the last
window solved before that execution terminated.

## Optional paired baseline study

The standalone study is not imported by `main.py` or the normal runners. Run it
only when needed:

```bash
python -m baselines.lane_study
```

It compares deterministic mean-trajectory STL with pdSTL for lane change and
lane merge using paired initial states and disturbances at three uncertainty
levels. Results, Wilson intervals, per-window certified hard intervals, and rate
figures are written under `outputs/lane_baseline/`. Override the trial count or
output location with `--trials` and `--output-dir`. Its only configuration is
`configs/experiments/lane_baseline.yaml`; `device: auto` selects CUDA when
PyTorch reports it available. The default is 600 planner runs, so first use
`--trials 1` to measure the printed live ETA on the target machine.

## Lane pipeline notebook

[`notebooks/lane_pipeline_demo.ipynb`](notebooks/lane_pipeline_demo.ipynb)
provides a lab-meeting walkthrough of both lane scenarios, from Gaussian beliefs
and the generated pdSTL task through receding-horizon execution, direct
certificate recomputation, final figures, and animations. It calls the normal
project runner and plotting code and reads the same editable scenario YAMLs.

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/lane_pipeline_demo.ipynb
```

Its figures and optional GIFs are isolated under `outputs/lane_notebook/`.

## Development checks

```bash
pip install -e ".[dev]"
ruff format --check src tests
flake8 src
MPLBACKEND=Agg pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
