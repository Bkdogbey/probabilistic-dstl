# Probabilistic differentiable STL

pdSTL evaluates Signal Temporal Logic specifications over uncertain state
trajectories. A model supplies a belief at each prediction step, the belief
evaluates atomic predicates as lower and upper probability bounds, and pdSTL
combines those bounds with Boolean and temporal operators.

```text
belief trajectory → predicate probability intervals → temporal formula trace
```

The implemented stochastic baseline is a precise Gaussian belief. Linear
dynamics propagate its mean and covariance. Threshold and affine half-space
events return exact bounds `[p_k, p_k]`; rectangles return a valid probability
enclosure from exact marginal interval probabilities. The generic `Belief` contract remains
`probability_bounds(predicate) -> [B, 2]`.

Planning follows this boundary:

```text
controls/current belief -> upstream rollout -> BeliefRollout
  -> BeliefTrajectory -> pdSTL -> planner objective
```

`BeliefRollout` requires only `belief_trajectory`. Its optional `nominal_trace`
and `aux` tensors supply costs and plots; they do not define belief semantics.
Set `w_dist`, `w_obs`, and `w_visit` to zero for objectives using only pdSTL and
control regularization. Enabled heuristic shaping requires a nominal trace.
`detach_diagnostics()` detaches these optional tensors and preserves the
original belief trajectory, including any gradient graph it retains.

Hard and differentiable evaluations are two versions of the same pdSTL / StoRI
semantics. With `scale=-1`, the formula returns the unsmoothed stochastic
robustness interval. With `scale=beta>0`, softplus replaces the conjunction's
lower clamp and smooth min/max replace temporal extrema. The planner uses the
differentiable lower semantics directly:

$$J(\mathbf u)=-w_\varphi\widetilde R^\downarrow_{\varphi,\beta}
  +w_uJ_u+w_{\Delta u}J_{\Delta u}.$$

Beta follows the existing geometric annealing schedule. Hard evaluations supply
monitoring, checkpoint selection, early stopping, and final verification; they
do not supply gradients in smooth mode. Neither the formula interval nor its
smooth approximation is claimed to be a whole-trajectory satisfaction
probability. Smooth values need not be valid probability bounds.

## Reach–Avoid example

The canonical double-slit scenario in `configs/scenarios/reach_avoid.yaml` keeps
all shaping weights (`w_dist`, `w_obs`, `w_visit`) at zero. Its pipeline is

```text
controls -> Gaussian mean/covariance rollout -> predicate probability intervals
         -> differentiable pdSTL lower semantics -> objective -> Adam update
returned controls -> hard pdSTL interval at planning origin 0
```

`HalfSpace(a, b)` in `pdstl.predicates` defines the event $a^T X\le b$ in any
state dimension. `GaussianBelief` evaluates it as
$\Phi((b-a^T\mu)/\sqrt{a^T\Sigma a})$, using the full covariance orientation.
A zero projected variance is evaluated as a deterministic closed inequality.

`InsideRectangle` and `OutsideRectangle` remain the specialized events for this
example. Exact Gaussian CDF differences give $p_x$ and $p_y$; the inside bounds
are $[\max(0,p_x+p_y-1),\min(p_x,p_y)]$. Outside bounds are the complement
$[1-U,1-L]$. These enclosures do not assume independent coordinates.

Running `run_reach_avoid(show=False, save=True)` from `planning.runners` produces
three figures in `outputs/`:

- `reach_avoid.png`: predicted belief mean, 95% joint covariance ellipses,
  workspace, three barrier blocks, goal, and start.
- `reach_avoid_probabilities.png`: two interval bands, for goal membership and
  combined safety at each prediction time.
- `reach_avoid_optimization.png`: differentiable lower robustness versus
  iteration, with the returned checkpoint marked and its hard interval reported.

The curve records each iteration's actual beta; annealing changes the evaluated
function, so the curve need not increase monotonically. The returned checkpoint
can precede the last iteration.

`PlanCandidate` exposes `smooth_lower`, `hard_interval`, `hard_lower`,
`hard_upper`, and `beta`. The saved `reach_avoid.pt` retains controls, covariances,
initial beliefs, `obstacle_traces`, `bounds_trace`, and `optimization_trace`
(iteration, beta, smooth lower, hard interval, control cost, and objective).
`history` remains the total objective history. Final `smooth_lower` is replayed
at `smooth_beta=beta_end`; the per-iteration values retain their original beta.
`interval_final` and `hard_interval` both report the final unsmoothed evaluation.

For comparisons, pass `show_initial=True` to `visualize_reach_avoid` or
`plot_reach_avoid`; `show_workspace=True` adds the workspace probability band to
`visualize_reach_avoid`. Controls and individual obstacle bounds remain available
as diagnostics without adding more default figures.

## Install and run

```bash
pip install -r requirements.txt
python src/main.py
```

Examples are selected with the numbered `skip_run` blocks in `src/main.py`.
Their model, predicate, confidence, and temporal-window settings live in
`configs/examples.yaml`. Set `show_plots: false` there for noninteractive runs.

## Package organization

```text
src/
  pdstl/
    base.py       Generic belief contracts and probability-bound validation
    operators.py  Boolean and temporal semantics
    predicates.py Spatial event geometry (half-spaces and rectangles)
  models/
    dynamics.py   Linear dynamics and bounded controls
    beliefs.py    Precise Gaussian beliefs, helpers, and trajectory factory
    rollouts.py   BeliefRollout and Gaussian rollout adapter
  planning/
    planner.py      Planner: how controls are optimized
    environment.py  Planning world and its pdSTL specification
    runners.py      Which experiment runs (altitude safety, reach-avoid, ...)
  baselines/      Deterministic STL comparison
  visualization/  Reusable plots
  main.py         AltitudeSafety and ReachAvoid demonstrations
configs/          Example and planning configuration
```

Import `GaussianBelief` and `create_gaussian_belief_trajectory` from
`models.beliefs`.
These names are no longer exported by `models.dynamics`. The enclosure demo
and its implementation have been removed; no replacement uncertainty model
is introduced.

## License

MIT. See LICENSE.
