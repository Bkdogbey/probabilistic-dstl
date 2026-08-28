"""Plotting for probability-bound traces.

This module only draws already-computed [lower, upper] traces; it does not
build predicates, formulas, sources, or Frechet bounds itself.
"""

import matplotlib.pyplot as plt
import torch

COLOR_PRIMARY = "tab:blue"
COLOR_SECONDARY = "tab:orange"
COLOR_COMBINED = "purple"


def plot_bounds(ax, time, bounds, label, color):
    """Shade [lower, upper] with a solid lower line and a dashed upper line.

    bounds is Tensor[1, T, 2] or Tensor[T, 2]; time is length T.
    """
    values = bounds[0] if bounds.dim() == 3 else bounds
    lower = values[:, 0].detach().cpu().numpy()
    upper = values[:, 1].detach().cpu().numpy()
    t = time.detach().cpu().numpy() if torch.is_tensor(time) else time

    ax.fill_between(t, lower, upper, color=color, alpha=0.15, label=label)
    ax.plot(t, lower, color=color, linestyle="-", marker="o", markersize=4)
    ax.plot(t, upper, color=color, linestyle="--", marker="o", markersize=4)


def plot_formula_bounds(title, series, show=True):
    """Plot one or more [lower, upper] traces together on one set of axes.

    series is an iterable of (time, bounds, label, color) tuples, each drawn
    with plot_bounds -- e.g. a predicate's raw bounds next to a temporal
    operator's reduced bounds, so you can see how the window changes them
    step by step. Nothing is written to disk; pass show=False (used by
    tests) to skip the plt.show() call.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    max_steps = 0
    for time, bounds, label, color in series:
        plot_bounds(ax, time, bounds, label, color)
        max_steps = max(max_steps, len(time))

    ax.set_title(title)
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Probability bound")
    ax.set_xticks(range(max_steps))
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if show:
        plt.show()
    plt.close(fig)
