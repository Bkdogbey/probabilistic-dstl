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
- `OnlineSource` — a source that grows one time step at a time via
  `.append({predicate: Tensor[B, 2]})`, for streaming use.
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

`src/main.py` is the single entry point for running the current examples.
It builds up one drone-altitude scenario in four steps:

1. **Predicate** — a single physically meaningful predicate, `altitude >= 50m`.
2. **Boolean** — combine two predicates (`altitude >= 50m`, `battery >= 20%`)
   with `&`, `|`, `~`.
3. **Always** — a bounded temporal operator alone: does the drone stay safe
   throughout a window?
4. **Always + Eventually** — the full mission ("stay safe and eventually
   land"), checked against independently hand-computed references and
   against an `OnlineSource` replay, then plotted.

Each example is its own function, toggled independently by flipping
`"run"` / `"skip"` in the `skip_run(...)` calls inside `main()` — the same
mechanism as `src/utils.py`'s `skip_run`. Edit `src/main.py` and flip a flag
to run just one example.

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
├── features/        # feature and predicate-probability extraction
├── models/          # example probability-bound inputs (temporal_examples.py)
├── pdstl/           # the pdSTL core: sources, predicates, Boolean/temporal operators
├── planning/        # trajectory optimization and receding-horizon control
├── visualization/   # plotting for probability-bound traces (probability_bounds.py)
└── main.py          # the single demonstration entry point
```

`baselines/`, `data/`, `datasets/`, `features/`, and `planning/` are
currently empty extension points reserved for future work; only `pdstl/`,
`models/`, `visualization/`, and `main.py` are implemented today.

## Testing

```bash
make test
# or
pytest -q
```
