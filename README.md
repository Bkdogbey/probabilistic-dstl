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

`src/main.py` is the single entry point for the selectable examples. The
offline examples first verify the probability-bound semantics; the sliding
and streaming examples then verify the bounded recurrent state.

1. **Boolean operators** (`run_boolean_example`) — two fixed probability
   intervals, `A = [0.60, 0.90]` and `B = [0.70, 0.95]`, combined with `~`,
   `&`, `|` and checked against the hand-computed Fréchet result. Numerical
   only, no model, no plot.
2. **Always** (`run_always_example`) — a seven-step Gaussian altitude belief
   with an ambiguous mean and known conditional standard deviation, evaluated
   over a configurable bounded interval.
3. **Eventually** (`run_eventually_example`) — a separate seven-step climbing
   belief using the same atomic-to-temporal probability-bound pipeline.
4. **Offline sliding Always** — eleven supplied atomic intervals evaluated
   over all complete `[0,2]` windows. A probability drop at `t=4` affects
   every three-entry window containing it, and the output recovers when it
   expires.
5. **Streaming Always** — the same trace is appended one time at a time. The
   temporal state grows to three entries, shifts thereafter, and produces the
   same bounds as the offline evaluation.
6. **Streaming Eventually** — repeats the incremental check with the union
   reduction used by Eventually.
7. **Streaming Always animation** — reveals one arrival at a time, highlights
   the retained and expired entries, writes out the current Fréchet calculation,
   and adds each completed temporal bound to the output history.

All experiment choices live in `configs/examples.yml`:

```yaml
show_plots: true

experiments:
  always:
    run: true
    threshold: 50.0
    interval: [0, 1]

  streaming_always:
    run: true
    interval: [0, 2]

  streaming_animation:
    run: false
    interval: [0, 2]
    frame_interval_ms: 900
    repeat: true
```

Set `run: false` to skip an experiment. Change its threshold or bounded
integer interval in the same entry; every example has an independent interval.
Set `show_plots: false` for a noninteractive run. A skipped experiment builds
nothing, evaluates nothing, and prints no experiment results.

For the clearest streaming view, set `streaming_animation.run: true` and set
the other experiment `run` values to `false`. Each frame shows the newest
atomic interval, the three retained entries, expired history, the current
Always calculation, and the output available at that arrival time.

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
├── visualization/   # offline temporal and streaming-state plots
└── main.py          # the single demonstration entry point
configs/
└── examples.yml     # run switches, thresholds, intervals, and plotting
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
