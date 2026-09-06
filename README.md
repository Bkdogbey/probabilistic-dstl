# Probabilistic dSTL

pdSTL evaluates bounded, discrete-time temporal specifications over **pointwise predicate probability intervals** using PyTorch. The core receives probabilities from an upstream model; it does not require a Gaussian distribution.

This branch, `pdstl-core`, implements the core revision from `opt-planning`. It preserves the matrix-based temporal register and the existing `scale` smoothing convention. Scenario and MPC integration are the next stage.

## Install and test

Python 3.10+ is required.

```bash
pip install -e '.[test,examples]'
python -m pytest
```

For core use only, `pip install -e .` installs `pdstl` and its PyTorch dependency. The `examples` extra provides dependencies for legacy source-tree helpers; existing scenarios have not yet been migrated to the revised output contract.

## Supply predicate probabilities

Each named predicate receives a floating tensor with shape `[batch, time, 2]`. The final axis stores `[lower, upper]`, with `0 <= lower <= upper <= 1`. All inputs share batch size, time length, dtype, and device. Exact probabilities use equal endpoints. Tensor inputs retain their autograd graph.

```python
import torch
from pdstl import Predicate, Always

p = torch.tensor([[0.95, 0.97, 0.96, 0.98]], requires_grad=True)
inputs = {"safe": torch.stack([p, p], dim=-1)}
spec = Always(Predicate("safe"), interval=[0, 2])

trace = spec(inputs)                  # [1, 2, 2]: two complete windows
origin = spec.robustness(inputs)      # [1, 1, 2]: [[ [.95, .95] ]]
assert spec.horizon == 2

smooth = spec.robustness(inputs, scale=20)
loss = -smooth[..., 0].mean()
loss.backward()                       # gradient reaches p
reported = spec.robustness(inputs)    # reevaluate the direct equations
```

A probability provider owns the justification for its bounds. A fully specified Gaussian and a scalar affine event can supply an exact probability; epistemic uncertainty or incomplete dependence information may justify an interval. A state's mean ± a standard-deviation multiplier is not, by itself, uncertainty about the Gaussian parameters.

## What the score means

For an atomic predicate, the interval bounds its pointwise event probability. Boolean composition uses Fréchet equations. Temporal Always and Eventually apply endpointwise minima and maxima. The resulting temporal interval is a **pointwise probabilistic robustness evaluation**, not generally a bound on whole-trajectory STL satisfaction probability.

For example, Always over two independent events of probability .95 has score .95, while the probability of their joint occurrence is .9025. Report that these are different quantities. Empirical trajectory satisfaction rate (SSR) requires evaluating complete sampled realizations separately.

See [the semantic contract](docs/semantics.md) for the equations, smoothing error, and time conventions.

## Time and smoothing

- Temporal intervals are explicit inclusive integer pairs `[a, b]`, with `0 <= a <= b`. `None`, infinity, and fractional endpoints are rejected.
- `spec.horizon` is the required lookahead. A trace of length `T` produces `[B, T - spec.horizon, 2]`. Fewer than `horizon + 1` predictions raises an error. Missing future samples are never replaced by a terminal value.
- `spec.robustness(inputs)` selects time zero; `keepdim=False` gives `[B, 2]`. Full traces always retain their time axis.
- `scale <= 0` evaluates the stated endpoint equations. `scale > 0` uses the existing unnormalized log-sum-exp smoothing. Smooth outputs can leave `[0,1]` or have crossed endpoints; they are optimization approximations. Reevaluate with `scale <= 0` for reporting or comparison with a requested score threshold.
- `spec.smoothing_error(scale)` gives an absolute endpoint-error bound relative to direct evaluation of that formula on the same inputs. It is not a probability confidence interval.

## Migration from opt-planning

The main interface is now `Predicate(name)` with supplied interval tensors. `GreaterThan`, `LessThan`, and the legacy belief containers remain available from their existing modules for migration. The Gaussian helper now handles zero variance and the trajectory wrapper preserves gradients, but its state-bound construction is still a legacy adapter whose probability interpretation must be justified upstream.

Conjunction now uses the Fréchet lower endpoint instead of a product. Until now combines witness and prefix with the stated Fréchet rule and includes the witness time in its left prefix. Both are semantic changes. Temporal outputs contain only complete windows, so existing plots and planners that expect `T` output positions must be updated before use. Unbounded and omitted intervals must be replaced with explicit finite requirements.

Always and Eventually retain the shift matrix `M`, insertion vector `b`, and reverse / recurrent / reverse evaluation. Until uses a separate prefix recurrence. `OnlineBeliefTrajectory` is an appendable list, not an incremental temporal monitor; each MPC solve must supply its own complete prediction trace.

## Project scope

`src/pdstl` contains the installed core. `src/models`, `src/planning`, `src/visualization`, and existing examples remain source-tree code awaiting integration. `src/pdstl/propagate.py` is a legacy NumPy/SciPy helper, not the differentiable prediction interface. No controller feasibility, closed-loop stability, or trajectory-level probability guarantee follows from this evaluator alone.

MIT license. Citation details for the associated work are forthcoming.
