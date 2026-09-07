# Probabilistic differentiable STL

pdSTL evaluates stochastic robustness intervals for Signal Temporal Logic over
belief trajectories. This branch uses Gaussian state beliefs for the introductory
examples and a general per-predicate interval input contract.

## Run the monitoring examples

With PyTorch, NumPy, SciPy, Matplotlib and PyYAML installed, run from the project root:

```bash
python src/main.py
```

`src/main.py` is a direct execution script with numbered `skip_run("run", ...)`
and `skip_run("skip", ...)` blocks, following the RA_L layout. Always and
Eventually run by default. Select experiments by editing these flags; there is
no main() wrapper or command-line dispatcher. Importing main executes its blocks;
import reusable modules from experiments, models or pdstl instead.

Set `show_plots: false` in `configs/stl_demos.yaml` for noninteractive execution.
Thresholds and integer step windows live in that same configuration. The
seven-point altitude fixtures have one-second spacing and standard deviation 1.5 m:

| Example | Mean altitude (m) | Threshold | Window |
| --- | --- | --- | --- |
| Always | 54, 53, 52, 51, 49, 52, 54 | 50 m | [0,1] |
| Eventually | 50, 51, 53, 54, 56, 57, 58 | 55 m | [0,1] |

Each runner prints atomic probabilities and complete temporal windows, and returns
`(time, mean, variance, atomic_trace, temporal_trace, figure)`. One shared visualizer
shows the state uncertainty, atomic probabilities and stochastic robustness.
`show=False` returns the figure without displaying it. Save and close it explicitly
when needed; monitoring does not save files automatically.

## Input and output

`Belief.probability_bounds(predicate)` returns `[B,2]` lower/upper endpoints.
`BeliefTrajectory` stores steps; predicates assemble `[B,T,2]` traces. Temporal
outputs are `[B,K,2]`, with K complete-window origins. Seven input points give six
origins for [0,1] and five for [1,2]. Output timestamps refer to origins, not window
starts. Insufficient input for one complete origin raises an error.

```python
from models.dynamics import always_altitude_example
from pdstl.operators import Always, GreaterThan
from utils import create_belief_trajectory

time, mean, variance = always_altitude_example()
beliefs = create_belief_trajectory(mean, variance, dtype=mean.dtype)
intervals = Always(GreaterThan(50.0), interval=[1, 2])(beliefs, scale=-1)
# [1,5,2], evaluated at origins 0 through 4
```

Use `PYTHONPATH=src` for this snippet. `models.beliefs.GaussianBelief` serves both
monitoring and planning. Its Gaussian affine probabilities are [p,p]; variance is
state uncertainty, not probability-interval width. `ProbabilityBelief` supports
already-computed intervals keyed by event name.

Boolean operators use Fréchet bounds; Always/Eventually apply endpointwise temporal
min/max. Temporal values are stochastic robustness, not general enclosures of
whole-trajectory satisfaction probability. Until uses an inclusive left prefix.
`scale <= 0` evaluates the defining equations. `scale > 0` uses the existing smooth
approximations, which can leave [0,1] or cross; report directly reevaluated robustness.

## Planning and organization

```text
src/
  main.py                  Direct experiment selection
  pdstl/                   Belief contract and operators
  models/beliefs.py        Shared Gaussian event probabilities
  models/dynamics.py       Integrators and altitude fixtures
  experiments/offline.py   Always/Eventually monitoring
  experiments/planning.py  Scenario runners
  planning/                Planner, environment, existing synthesis examples
  visualization/           Shared temporal and planning plots
configs/                   Monitoring and scenario parameters
outputs/                   Ignored generated results
```

Single-shot, MPC, and lane-change scenarios remain in skipped main.py blocks.
Normal/aggressive lane change share `run_lane_change(config_path=...)`.
`show=False` disables their presentation and live callbacks. New scenario caches
and animations go to outputs/, created only when saving. Old saved_data/*.pt
caches are ignored and are loaded only through an explicit load_from path.

The newer five optimization cases remain available through
`planning.examples.run_case` and `run_all`, with their configuration in
`configs/scenarios/examples.yaml`. Their visualization and tests were retained
when restoring the monitoring script. They are not launched by default.

This restoration preserves the newer planner implementation and the core equations.
Auxiliary geometry objectives, geometric probability approximations and MPC belief
assumptions still need separate review. Passing integration tests does not establish
closed-loop probabilistic guarantees.

## Verification

With pytest installed:

```bash
PYTHONPATH=src MPLBACKEND=Agg python -m pytest -q
```

CPU is the default. `PDSTL_DEVICE=cuda` opts planning into an available CUDA device.
Packaging/dependency metadata remain due for a separate cleanup.

## License

MIT. See LICENSE.
