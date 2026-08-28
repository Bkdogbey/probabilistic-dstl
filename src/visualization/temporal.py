"""Plotting for one standalone temporal-operator experiment.

Receives already-computed results only -- no predicates, formulas, Gaussian
probabilities, or Frechet bounds. Returns its Figure and never closes it;
pass show=False (used by tests) to skip plt.show().
"""

import matplotlib.pyplot as plt
import torch

COLOR_PRIMARY = "tab:blue"
COLOR_RESULT = "purple"


def _as_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def plot_temporal_example(model, atomic_bounds, temporal_bounds, formula_label, show=True):
    """Three stacked panels for one belief/formula pair.

    model is an AltitudeBelief (time/mean/std/threshold, each length T).
    atomic_bounds is Tensor[1, T, 2] with lower==upper (exact belief
    probability). temporal_bounds is Tensor[1, 1, 2], the single-window
    [lower, upper] result at anchor k=0.
    """
    fig, (ax_belief, ax_atomic, ax_result) = plt.subplots(3, 1, figsize=(7, 8))

    t, mean, std = _as_numpy(model.time), _as_numpy(model.mean), _as_numpy(model.std)
    ax_belief.plot(t, mean, color="black", marker="o", markersize=4, label="belief mean")
    ax_belief.fill_between(t, mean - std, mean + std, color="gray", alpha=0.2, label="mean +/- std")
    ax_belief.axhline(model.threshold, color="red", linestyle="--", linewidth=1, label=f"{model.threshold} m")
    ax_belief.set_title("Altitude belief")
    ax_belief.set_xlabel("Time step k")
    ax_belief.set_ylabel("Altitude (m)")
    ax_belief.set_xticks(t)
    ax_belief.grid(True, alpha=0.3)
    ax_belief.legend(loc="best", fontsize=8)

    p = _as_numpy(atomic_bounds[0, :, 0])
    ax_atomic.plot(t, p, color=COLOR_PRIMARY, marker="o", markersize=4)
    ax_atomic.set_title(f"P(altitude >= {model.threshold} m)")
    ax_atomic.set_xlabel("Time step k")
    ax_atomic.set_ylabel("Probability")
    ax_atomic.set_xticks(t)
    ax_atomic.set_ylim(0, 1.0)
    ax_atomic.grid(True, alpha=0.3)

    lower, upper = temporal_bounds[0, 0].tolist()
    ax_result.vlines(0, lower, upper, color=COLOR_RESULT, linewidth=10, alpha=0.7)
    ax_result.plot([0, 0], [lower, upper], color=COLOR_RESULT, marker="o", markersize=6)
    ax_result.set_xlim(-1, 1)
    ax_result.set_xticks([0])
    ax_result.set_xticklabels([formula_label])
    ax_result.set_ylim(0, 1.0)
    ax_result.set_ylabel("Probability bound")
    ax_result.set_title(f"{formula_label} = [{lower:.3f}, {upper:.3f}]")
    ax_result.grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
