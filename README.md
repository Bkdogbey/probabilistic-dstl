# Probabilistic differentiable STL

pdSTL evaluates Signal Temporal Logic specifications over uncertain state
trajectories. A model supplies a belief at each prediction step, the belief
evaluates atomic predicates as lower and upper probability bounds, and pdSTL
combines those bounds with Boolean and temporal operators.

```text
belief trajectory → predicate probability intervals → temporal formula trace
```

The implemented stochastic baseline is a precise Gaussian belief. Linear
dynamics propagate its mean and covariance, and each atomic predicate returns
equal probability bounds `[p_k, p_k]`. The generic `Belief` contract remains
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

Temporal outputs are stochastic robustness intervals; positive smoothing
scales produce optimization surrogates. They are not whole-trajectory
satisfaction probabilities.

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
