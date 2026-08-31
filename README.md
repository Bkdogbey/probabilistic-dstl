# Probabilistic dSTL (pdSTL)

pdSTL evaluates discrete-time Signal Temporal Logic formulas over
**probability bounds** rather than deterministic signals or point
probabilities. Every predicate, at every time step, is a tensor
`[lower, upper]` with `0 <= lower <= upper <= 1` — an enclosure of the true
satisfaction probability, not the probability itself. An exact probability
`p` is just the degenerate interval `[p, p]`.

Because the joint dependence between predicates is generally unknown, Boolean
and temporal composition use **Fréchet bounds**: the tightest enclosure
valid for *any* dependence structure between the combined events, with no
independence assumption. Hard evaluation (the default) returns this
certified enclosure directly.

## Core API

- `ProbabilitySource` — the abstract input contract: `bounds(predicate, time)`
  returns `Tensor[B, 2]`, `len(source)` returns the number of available time
  steps.
- `OfflineSource` — a complete `{predicate: Tensor[B, T, 2]}` trace per
  predicate, known up front.
- `Predicate(name)` — an atomic formula; its bounds come directly from a
  `ProbabilitySource`.
- `Not`, `And`, `Or` — pointwise Boolean composition (`~`, `&`, `|`) under
  Fréchet bounds.
- `Always(child, (a, b))`, `Eventually(child, (a, b))` — bounded temporal
  operators over a fixed discrete window, also under Fréchet bounds.

Every formula is a `torch.nn.Module`. Calling it on a source —
`formula(source)` — returns `Tensor[B, T, 2]`.

## Hard vs. smooth evaluation

```python
formula(source)                          # hard (default): certified bound
formula(source, smooth=True, beta=20.0)  # smooth: optimization surrogate
```

- **Hard** (`smooth=False`, the default) computes the exact Fréchet
  reduction. This is the number to report or monitor: it is a valid
  probability interval.
- **Smooth** (`smooth=True`) replaces the hard `min`/`max`/clamp reductions
  with `softplus`/`logsumexp` so gradients reach the probability source even
  where the hard reduction is flat. **It is not a certified probability
  bound** — it is a differentiable surrogate for use inside an optimization
  loop, and it only approaches the hard result as `beta` increases. Always
  rerun with `smooth=False` to get the number you actually report.

Because every operator is a standard `torch.nn.Module` built from
differentiable tensor operations, gradients flow through smooth evaluation
back to the tensors supplied by the `ProbabilitySource` via ordinary PyTorch
autograd (`.backward()`).

## Installation

```bash
pip install -e ".[dev]"
```

This installs the package (`pdstl`) plus `pytest`, `ruff`, and `matplotlib`
for development. `import pdstl` then works from anywhere, not just this
repository.

## Running the demonstration

`src/main.py` is the single entry point and contains exactly four independent
demonstration blocks:

1. **Boolean probability bounds** (`run_boolean_example`) — two fixed
   probability intervals, `A = [0.60, 0.90]` and `B = [0.70, 0.95]`, combined
   with `~`, `&`, `|` and checked against the hand-computed Fréchet result.
   Numerical only, no model, no plot.
2. **Offline temporal operators** (`run_always_example` and
   `run_eventually_example`) — evaluates separate seven-step Gaussian altitude
   beliefs with configurable thresholds and bounded intervals.
3. **Streaming bounded monitor** (`run_streaming_always_example` or
   `run_streaming_always_animation`) — streams Always through one persistent
   temporal state and verifies every online output against the offline result.
   Animation is an option of this example, not a separate experiment.
4. **Safe Until goal** (`run_until_example`) — evaluates `safe U[1,2] goal`.
   Each possible goal time
   forms a candidate requiring safety through the goal step; candidate
   intervals are unioned and checked in both offline and streaming modes.

Choose which demonstrations run by changing each literal `"run"` or `"skip"`
flag in `src/main.py`. The shipped default runs only the Boolean demonstration.
`configs/examples.yml` contains numerical and presentation parameters only:

```yaml
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
```

Set `show_plots: false` for a noninteractive run. For the streaming block,
`animate: true` selects the existing animation runner and `animate: false`
selects the static runner; both use the same interval and verify online results
against the complete offline trace. A skipped block builds and evaluates
nothing.

For the temporal examples, the uncertain altitude follows

```text
Z_t | mu_t ~ Normal(mu_t, std_t^2),  mu_t in [mean_lower_t, mean_upper_t].
```

For the event `Z_t >= h`, monotonicity in the Gaussian mean gives atomic
bounds derived with the normal CDF:

```text
p_lower_t = Phi((mean_lower_t - h) / std_t)
p_upper_t = Phi((mean_upper_t - h) / std_t)
```

These bounds are computed from the belief, never hand-authored. Applying
Always or Eventually then produces temporal probability bounds using Fréchet
semantics because dependence between events at different times is unknown.
The three plot panels expose this full progression: admissible Gaussian means,
atomic lower/upper probabilities, and temporal lower/upper probabilities at
every valid STL anchor.

The streaming examples use `OnlineSource` as the growing input store and pass
only the newest atomic interval through `TemporalOperator.step()`. For
`[0,2]`, the first output becomes available at arrival `t=2`; the state then
keeps the most recent three entries. The example compares every incremental
output against the complete `OfflineSource` result.

For bounded strong Until,

```text
safe U[a,b] goal
```

the goal must occur at some offset `j` in `[a,b]`, while safety holds at every
offset from `0` through `j`, including the goal time. This is the overlapping
Until convention used by STLCG. The implementation builds one candidate event
for each possible `j`, applies the finite-intersection Fréchet bounds within
each candidate, and then applies the finite-union bounds across candidates.
Every candidate shares the safety prefix from `0` through `a`, so that common
prefix also tightens the final upper bound when there is more than one
candidate. The streaming cell
retains named left and right windows and produces the same result as offline
evaluation once the right edge `b` has arrived.

```bash
python src/main.py
# or, equivalently:
make demo
```

## Example

```python
import torch
from pdstl import Always, OfflineSource, Predicate

safe = Predicate("safe")
source = OfflineSource({safe: torch.tensor([[0.9, 0.95], [0.85, 0.9], [0.8, 0.9]]).unsqueeze(0)})

always_safe = Always(safe, (0, 2))
always_safe(source)                          # hard, certified bound
always_safe(source, smooth=True, beta=20.0)  # smooth surrogate for optimization
```

## Project structure

```text
src/
├── baselines/       # deterministic STL and comparison methods
├── data/            # data loading and preprocessing
├── datasets/        # experiment datasets
├── experiments/     # offline and streaming example orchestration
├── features/        # feature and predicate-probability extraction
├── models/          # Boolean, Gaussian-belief, and streaming inputs
├── pdstl/           # the pdSTL core: sources, predicates, Boolean/temporal operators
├── planning/        # trajectory optimization and receding-horizon control
├── visualization/   # offline, streaming-state, mission, and Until plots
└── main.py          # the single demonstration entry point
configs/
└── examples.yml     # thresholds, intervals, animation, and plotting
```

`baselines/`, `data/`, `datasets/`, `features/`, and `planning/` are
currently empty extension points reserved for future work; `experiments/`,
`pdstl/`, `models/`, `visualization/`, and `main.py` implement the current
demonstrations.

## Testing

```bash
make test
# or
pytest -q
```
