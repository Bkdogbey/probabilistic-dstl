"""Minimal pdSTL usage: predicates, Boolean and temporal operators, hard vs
smooth evaluation, and a streaming (online) source.

Run with:
    python examples/basic.py
"""

import torch

from pdstl import Always, OfflineSource, OnlineSource, Predicate

safe = Predicate("safe")
goal = Predicate("goal")

# Each predicate's trace is Tensor[B, T, 2] = [lower, upper] probability bounds.
source = OfflineSource(
    {
        safe: torch.tensor([[0.9, 0.95], [0.85, 0.9], [0.8, 0.9]]).unsqueeze(0),
        goal: torch.tensor([[0.6, 0.7], [0.65, 0.75], [0.7, 0.8]]).unsqueeze(0),
    }
)

conjunction = safe & goal
print("safe AND goal, per step (Frechet bounds, no independence assumed):")
print(conjunction(source))

# A bounded temporal operator: "safe holds at every step in [0, 2]".
always_safe = Always(safe, (0, 2))

# Hard evaluation (the default, smooth=False): the certified probability
# enclosure -- 0 <= lower <= upper <= 1 is guaranteed.
hard = always_safe(source)
print(f"\n{always_safe} -- hard, certified bound:")
print(hard)

# Smooth evaluation: a DIFFERENTIABLE OPTIMIZATION SURROGATE, not a certified
# probability interval. It may fall outside the true enclosure at finite beta
# and only approaches the hard result as beta grows. Use it to get gradients
# during optimization, then rerun with smooth=False (the default) to get the
# certified bound you actually report.
smooth = always_safe(source, smooth=True, beta=20.0)
print(f"\n{always_safe} -- smooth SURROGATE (not certified), beta=20:")
print(smooth)

# OnlineSource grows one time step at a time; re-evaluating after each append
# reflects only the steps seen so far.
print("\nOnlineSource: append one step at a time and re-evaluate")
online = OnlineSource()
for step, (lower, upper) in enumerate([(0.9, 0.95), (0.85, 0.9), (0.8, 0.9)]):
    online.append({safe: torch.tensor([[lower, upper]])})
    out = always_safe(online)
    print(f"  after step {step}: shape={tuple(out.shape)}  {out.tolist()}")
