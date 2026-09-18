# Probabilistic differentiable STL

pdSTL evaluates Signal Temporal Logic formulas over uncertain trajectories.
A belief supplies atomic probability bounds; Boolean and temporal operators
combine them. The generic contract is probability_bounds(predicate) -> [B, 2].
The included Gaussian model evaluates threshold, half-space, and rectangle events.
Rectangular enclosures do not assume independent coordinates.

## Planning

The execution path is:

    config -> environment and specification -> stochastic rollout
           -> smooth lower pdSTL score -> optimize controls
           -> final hard interval -> PlanResult

Planner(dynamics, horizon, config=None) accepts rollout(v) -> BeliefRollout
and an explicit spec in optimize_window. Only the belief trajectory is required;
nominal and auxiliary traces are optional diagnostics.

The only objective is:

    -w_phi * smooth_lower + w_u * control_effort + w_du * control_smoothness

Effort is the sum of squared controls. Smoothness is the sum of squared successive
differences plus the squared first control. The initial guess is in physical
control units; omission means zero controls. Dynamics bounds controls with tanh.

The planner differentiates spec.smooth_lower(trajectory, beta), using a geometric
beta schedule. It returns the final optimizer update, reports its smooth score
at beta_end, and evaluates spec.probability_interval(trajectory) for validation.
Hard scores never select checkpoints. Optional hard-threshold stopping is
disabled by default (alpha: null).

Smooth values need not be probability bounds. The hard interval is a pdSTL/StoRI
evaluation, not an exact whole-trajectory satisfaction probability.

PlanResult contains controls, rollout, smooth_lower, hard_interval, loss_history.
History records the loss before each optimizer update at that iteration's beta.
Optional iteration observers receive post-update plans. Returned predictions
retain no optimization graph.

## Receding horizon

Planner.run_receding_horizon is the only MPC loop:

    current state/belief -> rollout and local specification -> optimize horizon
                        -> execute first control -> update -> shift -> repeat

Supply make_rollout(state, step), make_spec(state, step),
execute(state, control, step), a pure is_done(state, step), and max_steps.
Execution returns a new state. Warm starts drop the executed control and
repeat the last control at the horizon tail.

MPCResult contains states, applied_controls shaped [steps, control_dim],
window_plans, and stopped_reason (goal_reached or max_steps).
States include the initial state, including when the loop executes zero controls.

## Scenarios and outputs

Reach-avoid and MPC share the workspace/goal/obstacles rectangular schema,
with configurable name, x, y, and optional style fields. The default has two
blocks forming one gap. Its specification is:

    G[1,H](inside workspace AND outside every obstacle)
    AND F[0,H](inside goal)

Lane merge preserves its moving vehicle, local goal/workspace rules, moment
predicate, and consecutive-step success criterion. It uses the canonical loss,
so numerical trajectories can differ from the former heuristic optimizer.
Lane settings live in the lane scenario YAML.

Public runners: run_reach_avoid, run_mpc, run_lane_change, run_altitude_safety.
All accept show, save, and verbose. With show=False and save=False, no plotting
diagnostics or output files are made.

Reach-avoid saves a structured result and three figures: predicted trajectory,
event probability intervals, and optimization loss. Altitude can animate
optimizer updates. MPC and lane runs can save trajectory figures and animations.
Visualization derives its data on demand rather than extending result schemas.

Save with torch.save; load trusted result files with
torch.load(path, weights_only=False). Old dictionaries and solve() dispatch
are intentionally unsupported.

## Install and run

    pip install -r requirements.txt
    python src/main.py

Select altitude and reach-avoid using the skip_run blocks in src/main.py.
Set show_plots: false in configs/examples.yaml for noninteractive runs.
Additional scalar demonstrations are in planning.examples.

## Remaining planning files

| File | Purpose |
| --- | --- |
| environment.py | Geometry, config parsing, reach-avoid/lane specification builders |
| planner.py | One optimizer, structured results, one generic MPC loop |
| runners.py | Setup, execution rules, saving and visualization orchestration |
| examples.py | Five scalar formula cases and scalar one-shot/MPC reach demonstrations |
| __init__.py | Minimal public exports |

The scenario subpackage, logging wrappers, alternative benchmark, and obsolete
single-shot runner/configuration have been removed.

Other packages: pdstl owns semantics and predicates; models owns beliefs,
dynamics, and rollouts; visualization owns plots and animations; baselines owns
the deterministic STL comparison. Existing Gaussian-moment predicates now live
in pdstl/predicates.py; generic geometric events retain their belief-independent
probability contract.

## License

MIT. See LICENSE.
