# Probabilistic dSTL

> *"I am 94.7% sure the robot won't crash. Probably."*

**pdSTL** is a Python library for evaluating and optimizing [Signal Temporal Logic (STL)](https://en.wikipedia.org/wiki/Signal_temporal_logic) specifications over **probabilistic (Gaussian belief) trajectories** — because the real world is uncertain and your specs shouldn't pretend otherwise.

Instead of asking *"does the robot always stay in the lane?"*, pdSTL asks the more honest question: ***"what is the probability that the robot always stays in the lane?"***

---

## What is this?

STL lets you write temporal requirements like:

```
□[1s, 5s] (x ≥ 50)      # "x must always be ≥ 50 between 1 and 5 seconds"
◇[0, 10] (goal reached)  # "reach the goal within 10 seconds"
```

Classical STL checks these against **deterministic** signals. But real robots live in an uncertain world — sensors are noisy, dynamics are approximate, wind exists.

**pdSTL** propagates Gaussian uncertainty through STL operators, giving you a **satisfaction probability** for every spec at every timestep. You can then use that probability as an objective to optimize trajectories that are *robustly safe* under uncertainty.

---

## Features

- **Probabilistic STL evaluation** — compute P(spec satisfied) for Gaussian belief trajectories
- **Gradient-based motion planning** — maximize satisfaction probability via PyTorch autograd
- **MPC (Receding Horizon)** — roll the planner forward in real time
- **Lane change scenarios** — dodge moving obstacles while staying in the lane, stochastically
- **Pluggable belief system** — bring your own `Belief` subclass; Gaussian is just the default

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/iHuman-Lab/probabilistic-dstl
cd probabilistic-dstl
pip install -e .

# 2. Run the example cases (non-interactive; plots land in outputs/)
python src/main.py --all
python src/main.py --case corridor   # one case
python src/main.py --case mpc        # MPC consistency check
python src/main.py --list            # available cases
```

### Evaluate a spec over a belief trajectory

```python
import numpy as np
from models.dynamics import linear_system, sinusoidial_input
from pdstl.operators import Always, GreaterThan
from utils import create_belief_trajectory, to_steps

t = np.linspace(0, 10, 100)
mean, var = linear_system(a=0.01, b=1.0, g=2.0, q=2.5,
                          mu=50.0, P=0.15, t=t,
                          control_func=sinusoidial_input)

beliefs = create_belief_trajectory(mean, var)

phi  = GreaterThan(threshold=50.0)
spec = Always(phi, interval=to_steps([1, 2], t))

trace = spec(beliefs)   # [B, K, 2] stochastic robustness interval

# `trace` holds only origins whose window is complete, so K < len(t): a
# bounded operator over [a, b] consumes b steps of lookahead. Endpoints are
# [lower, upper]; the Gaussian atom's own values are exact probabilities.
```

---

## Examples

Five Gaussian single-integrator cases share one runner
([src/planning/examples.py](src/planning/examples.py)); parameters and the model
assumptions live in [configs/scenarios/examples.yaml](configs/scenarios/examples.yaml).

| Case | Specification | Purpose |
|---|---|---|
| `always` | □[1,H](x ≥ c) | Raise the least favourable future atom probability |
| `eventually` | ◇[a,b](x ≥ c) | Reach a target inside a window |
| `corridor` | □[1,H](x ≥ c₁ ∧ x ≤ c₂) | Boolean interval composition inside a temporal operator |
| `until` | (x ≤ cs) U[a,b] (x ≥ cg) | Inclusive witness with the required left prefix |
| `nested` | ◇[a,b](□[0,d](x ≥ c)) | Arrival then persistence; nested lookahead |

`python src/main.py --all` runs all five. The planning scenarios (single-shot,
MPC, lane change) live in [src/planning/runners.py](src/planning/runners.py) and
are invoked explicitly; nothing here starts them automatically.

**Interpreting the output.** The temporal output is a *stochastic robustness*
interval, not a whole-trajectory satisfaction probability. Only the Gaussian
atom values are probabilities. Plots show mean ± σ as a **state uncertainty
band** — not a probability interval and not a guaranteed tube. A formula trace
covers only origins with a complete window, so its series is shorter than the
state trace and the remainder is shown as absent rather than padded. Under a
positive smoothing `scale` the optimiser's score is an approximation that may
leave [0,1]; reported intervals always come from a direct (`scale ≤ 0`)
re-evaluation of the same formula.

---

## Project Structure

```
src/
├── pdstl/          # Core: Belief base classes, STL operators, propagation
├── models/         # Dynamical systems (linear, double integrator, etc.)
├── planning/       # Gradient-based planner, MPC runner, environments
├── visualization/  # Robustness plots, animations, live MPC callbacks
├── baselines/      # Deterministic STL baseline for comparison
└── main.py         # Named-case entry point
configs/            # YAML configs for scenarios and hyperparameters
outputs/            # Generated plots (git-ignored)
saved_data/         # Cached optimization results (git-ignored, regenerable)
```

---

## Device Configuration

The library defaults to **CPU** (some machines expose CUDA even when it can't initialize). To use a GPU:

```bash
PDSTL_DEVICE=cuda python src/main.py --all
# or
PDSTL_USE_CUDA=1 python src/main.py --all
```

---

## Requirements

- Python 3.8+
- PyTorch (for autograd-based planning)
- NumPy, PyYAML, python-dotenv

```bash
pip install -r requirements.txt
```

---

## Citation

If this library is useful in your research, please consider citing the associated work (details forthcoming).

---

## License

MIT — do whatever you want, but don't blame us if the robot crashes. (We did say *probabilistic*.)
