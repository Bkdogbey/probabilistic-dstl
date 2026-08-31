# Probabilistic dSTL (pdSTL)

**pdSTL** is a PyTorch implementation of bounded-time Signal Temporal Logic
over probability intervals. Instead of requiring a deterministic state signal,
the current monitor accepts lower and upper bounds on the probability that
each atomic predicate is satisfied:

~~~text
[lower, upper], where 0 <= lower <= upper <= 1.
~~~

An exact probability p is represented by [p, p]. Boolean and temporal
operators combine these intervals with Fréchet bounds, so the hard semantics
do not introduce an independence assumption when dependence between events is
unknown.

This RA_L branch is under active development. Its present scope is the
probability-bound monitoring graph: atomic inputs, Boolean operators, bounded
temporal operators, and matching offline and streaming evaluation. Trajectory
optimization has not yet been added to this branch.

## What is implemented

- A small ProbabilitySource interface for externally computed atomic
  satisfaction-probability bounds.
- Complete offline traces through OfflineSource.
- Incremental inputs through OnlineSource.
- Boolean Not, And, and Or using Fréchet bounds.
- Bounded Always, Eventually, and strong Until.
- One temporal step mechanism shared by offline and streaming evaluation.
- Standard torch.nn.Module formula composition.
- Hard probability-bound semantics for monitoring and reporting.
- An experimental smooth mode for studying gradient-based optimization.

The core does not require a particular state distribution. The included drone
example uses an ambiguous Gaussian altitude belief, while the streaming
examples supply probability intervals directly.

## Quickstart

~~~bash
git clone https://github.com/Bkdogbey/probabilistic-dstl.git
cd probabilistic-dstl
git switch RA_L
pip install -e ".[dev]"
python src/main.py
~~~

The package requires Python 3.10 or newer. Runtime dependencies are PyTorch
and PyYAML; the development installation also includes pytest, Ruff, and
Matplotlib.

## Basic use

~~~python
import torch

from pdstl import Always, OfflineSource, Predicate

safe = Predicate("safe altitude")
atomic_bounds = torch.tensor(
    [[[0.90, 0.95], [0.85, 0.90], [0.80, 0.90]]]
)
source = OfflineSource({safe: atomic_bounds})

formula = Always(safe, (0, 2))
temporal_bounds = formula(source)
~~~

temporal_bounds has shape [batch, valid_anchors, 2]. Its last dimension
contains the lower and upper satisfaction-probability bounds.

## Demonstrations

src/main.py is the single demonstration entry point. It contains four
independent skip_run blocks:

| # | Demonstration | Purpose |
|---:|---|---|
| 1 | Boolean probability bounds | Verifies Not, And, and Or against hand calculations. |
| 2 | Offline temporal operators | Converts Gaussian altitude beliefs into atomic bounds and evaluates Always and Eventually. |
| 3 | Streaming bounded monitor | Shows an Always window filling and sliding, then checks each online output against offline evaluation. |
| 4 | Safe Until goal | Builds and unions the candidate events for bounded strong Until, offline and online. |

Choose what runs by editing the literal flag beside each block in
src/main.py:

~~~python
with skip_run("run", "Boolean probability bounds") as check, check():
    run_boolean_example()

with skip_run("skip", "Offline temporal operators") as check, check():
    ...
~~~

The shipped default runs only the Boolean example. Change "skip" to "run"
for the experiment you want to inspect.

configs/examples.yml contains experiment parameters only:

~~~yaml
show_plots: true

experiments:
  offline_temporal:
    always:
      threshold: 50.0
      interval: [0, 1]
    eventually:
      threshold: 55.0
      interval: [0, 1]

  streaming:
    interval: [0, 2]
    animate: true
    frame_interval_ms: 900
    repeat: true

  until:
    interval: [1, 2]
~~~

Set show_plots to false for a noninteractive run. In the streaming example,
animate: true selects the animation and animate: false selects the static
view. Both paths verify the streaming outputs against the complete offline
result.

## Mechanism

At time t, a probability source supplies [p_lower_t, p_upper_t] for the event
that an atomic predicate is satisfied. A Predicate retrieves that interval,
and the formula graph propagates it through Boolean and temporal operators.

For example, Always(child, (a, b)) evaluates every complete window with the
finite-intersection Fréchet bounds

~~~text
lower = max(0, sum(child_lower) - (window_size - 1))
upper = min(child_upper)
~~~

The streaming temporal state retains the most recent b + 1 atomic intervals
so that expired entries can leave a sliding window. Offline evaluation unrolls
the same step transition over a complete trace; it is not a separate
semantics.

See [docs/mechanism.md](docs/mechanism.md) for the complete mathematical and
architectural walkthrough, including tensor shapes, Boolean propagation,
temporal windows, recurrent state, and current limitations.

## Hard and smooth modes

~~~python
hard_bounds = formula(source)
smooth_score = formula(source, smooth=True, beta=20.0)
~~~

Hard mode is the current monitoring semantics and returns the implemented
Fréchet enclosure. Smooth mode replaces hard reductions with softplus and
log-sum-exp operations so gradients can propagate through more branches.


## Project structure

~~~text
src/
├── pdstl/          # Probability sources and STL formula operators
├── models/         # Gaussian-belief and supplied probability traces
├── experiments/    # Offline, streaming, composed, and Until runners
├── visualization/  # Temporal plots and streaming animation
└── main.py         # Four selectable demonstrations
configs/
└── examples.yml    # Thresholds, intervals, and display settings
docs/
└── mechanism.md    # Mathematical and architectural walkthrough
tests/              # Semantics, gradients, examples, and online/offline checks
~~~

Some top-level source directories remain as placeholders for later planning,
data, feature, and baseline work. They are not part of the implemented
pipeline described above.

## Testing

~~~bash
make test
make lint
~~~

or directly:

~~~bash
pytest -q
ruff check .
~~~

## Development status

The next stage is to validate the differentiable path end to end:

~~~text
controls -> dynamics -> belief -> atomic bounds -> smooth pdSTL score -> loss
~~~

The optimization objective, control model, and planner will be added only
after the smooth operator behavior has been selected and verified against the
hard probability bounds.

## License

This project is released under the MIT License.
