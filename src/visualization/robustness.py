"""Plots for pdSTL example cases.

Traces are indexed from origin 0 and contain only origins whose window is
complete (see pdstl.operators), so a formula trace of length K is shorter than
the state trace. K values are drawn at the first K origin timestamps -- never
shifted by the interval's lower bound, and never padded with repeated terminal
values. Origins past K have no evaluation and are shown as absent.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Tableau 10
_BLUE = "#1f77b4"
_RED = "#d62728"
_GREEN = "#2ca02c"
_GRAY = "#7f7f7f"

# Endpoints closer than this are coincident for display purposes; Gaussian
# atoms return [p, p], whose fill between endpoints would be invisible.
_COINCIDENT = 1e-9


def _to_numpy(trace):
    """Return a probability-bound trace as numpy [K, 2], whatever its length."""
    if isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()
    trace = np.asarray(trace)

    if trace.ndim == 3:  # [B, K, 2] -> first batch element
        trace = trace[0]
    elif trace.ndim == 1:  # exact scalar series -> coincident endpoints
        trace = np.stack([trace, trace], axis=-1)

    if trace.ndim != 2 or trace.shape[-1] != 2:
        raise ValueError(f"Expected a [K, 2] bound trace, got {trace.shape}")
    return trace


def _finish(fig, save_path=None, show=False):
    """Save and/or display, so runs can be fully non-interactive."""
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def _draw_bounds(ax, time, bounds, label_lower, label_upper):
    """Draw a [K, 2] bound trace at the first K origin timestamps.

    Coincident endpoints (an exactly known probability) are drawn as a single
    marked series rather than a zero-height band that would be invisible.
    """
    k = len(bounds)
    t = np.asarray(time)[:k]
    lower, upper = bounds[:, 0], bounds[:, 1]
    coincident = bool(np.all(np.abs(upper - lower) <= _COINCIDENT))

    # A single origin has no line to draw, so always mark the points.
    marker = "o"

    if coincident:
        ax.plot(t, lower, color=_BLUE, lw=1.8, marker=marker, ms=5,
                label=f"{label_lower} = {label_upper}")
    else:
        ax.fill_between(t, lower, upper, alpha=0.25, color=_GREEN)
        ax.plot(t, lower, color=_BLUE, lw=1.8, marker=marker, ms=4, label=label_lower)
        ax.plot(t, upper, color=_RED, lw=1.8, marker=marker, ms=4, label=label_upper)

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return k


def _mark_absent(ax, time, k):
    """Shade origins that have no complete window, so they read as absent."""
    time = np.asarray(time)
    if k >= len(time):
        return
    # Start midway to the next step so the last valid origin's marker stays clear.
    start = (time[k - 1] + time[k]) / 2 if k else time[0]
    ax.axvspan(start, time[-1], color=_GRAY, alpha=0.10)
    ax.text(
        (start + time[-1]) / 2,
        0.5,
        "no complete\nwindow",
        ha="center", va="center", fontsize=8, color=_GRAY,
    )


def plot_case(
    time,
    formula_trace,
    mean_trace=None,
    var_trace=None,
    lower_trace=None,
    upper_trace=None,
    predicate_traces=None,
    thresholds=None,
    formula_str=None,
    title=None,
    figsize=(9, 8),
    save_path=None,
    show=False,
):
    """Three-panel view of one case: state, atom probabilities, formula interval.

    time             : [T] physical timestamps of prediction steps 0..T-1
    formula_trace    : [B, K, 2] or [K, 2] formula bounds, K <= T
    mean_trace       : [T] mean of the constrained state component
    var_trace        : [T] variance of that component
    lower_trace, upper_trace : [T] explicit descriptor bounds of the state
        component, when the belief is a propagated enclosure rather than a
        point mean. Takes precedence over mean_trace for panel (a); var_trace
        may still be given alongside them and is drawn as a separate,
        visually distinct residual-spread envelope, not folded into the
        descriptor band.
    predicate_traces : {label: [B, T, 2]} atomic probabilities
    """
    time = np.asarray(time)
    formula = _to_numpy(formula_trace)

    n_panels = 3 if predicate_traces else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    ax_state = axes[0]
    ax_pred = axes[1] if predicate_traces else None
    ax_form = axes[-1]

    # (a) state and its uncertainty
    if lower_trace is not None and upper_trace is not None:
        lower, upper = np.asarray(lower_trace), np.asarray(upper_trace)
        if var_trace is not None:
            sigma = np.sqrt(np.maximum(np.asarray(var_trace), 0.0))
            ax_state.fill_between(
                time, lower - sigma, upper + sigma, alpha=0.12, color=_GRAY,
                zorder=1, label=r"$\pm$ residual std",
            )
        ax_state.fill_between(
            time, lower, upper, alpha=0.25, color=_BLUE, zorder=2,
            label="descriptor bounds [lower, upper]",
        )
        ax_state.plot(
            time, (lower + upper) / 2, color=_BLUE, lw=1.8, marker="o", ms=3,
            zorder=3, label="descriptor midpoint",
        )
    elif mean_trace is not None and var_trace is not None:
        mean = np.asarray(mean_trace)
        sigma = np.sqrt(np.maximum(np.asarray(var_trace), 0.0))
        ax_state.fill_between(
            time, mean - sigma, mean + sigma, alpha=0.25, color=_BLUE,
            label=r"state uncertainty band $\mu \pm \sigma$",
        )
        ax_state.plot(time, mean, color=_BLUE, lw=1.8, marker="o", ms=3, label=r"$\mu(t)$")

    if (mean_trace is not None or lower_trace is not None) and thresholds is not None:
        for th in np.atleast_1d(thresholds):
            ax_state.axhline(float(th), color=_RED, ls="--", lw=1.3)
        ax_state.plot([], [], color=_RED, ls="--", lw=1.3, label="threshold")

    ax_state.set_ylabel("state $x$")
    ax_state.set_title("(a) Predicted state", loc="left", fontweight="bold")
    ax_state.legend(fontsize=8, loc="best", framealpha=0.95)
    ax_state.grid(True, alpha=0.3)

    # (b) atomic event probabilities
    if ax_pred is not None:
        for label, tr in predicate_traces.items():
            bounds = _to_numpy(tr)
            t = time[: len(bounds)]
            ax_pred.plot(t, bounds[:, 0], lw=1.6, marker="o", ms=3, label=label)
        ax_pred.set_ylabel("probability")
        ax_pred.set_ylim(-0.05, 1.05)
        ax_pred.set_title(
            "(b) Atomic event probability (exact, endpoints coincide)",
            loc="left", fontweight="bold",
        )
        ax_pred.legend(fontsize=8, loc="best", framealpha=0.95)
        ax_pred.grid(True, alpha=0.3)

    # (c) formula interval at its valid origins only
    k = _draw_bounds(ax_form, time, formula, "lower", "upper")
    _mark_absent(ax_form, time, k)
    ax_form.set_ylabel("stochastic robustness")
    ax_form.set_xlabel("time [s]")
    ax_form.set_title(
        f"(c) Formula interval at {k} valid origin{'s' if k != 1 else ''}",
        loc="left", fontweight="bold",
    )
    ax_form.legend(fontsize=8, loc="best", framealpha=0.95)

    heading = title or formula_str
    if heading:
        fig.suptitle(heading, fontsize=11, fontweight="bold")

    _finish(fig, save_path, show)
    return fig, axes


def plot_synthesis(
    time,
    initial,
    optimized,
    objective_history,
    thresholds=None,
    title=None,
    figsize=(10, 7),
    save_path=None,
    show=False,
):
    """Initial vs optimized behaviour plus a compact objective history.

    `initial` and `optimized` are dicts with a ``formula`` key (the directly
    evaluated bound trace) plus either ``mean``/``var`` (a point-mass belief,
    band drawn as mean +- sigma) or ``lower``/``upper`` (a propagated
    descriptor enclosure, band drawn from the bounds directly; ``var`` may
    still be given alongside them for a separate residual-spread envelope).
    """
    time = np.asarray(time)
    fig, ((ax_state, ax_form), (ax_obj, ax_blank)) = plt.subplots(
        2, 2, figsize=figsize
    )
    ax_blank.axis("off")

    for run, color, name in ((initial, _GRAY, "initial"), (optimized, _BLUE, "optimized")):
        if "lower" in run and "upper" in run:
            lower, upper = np.asarray(run["lower"]), np.asarray(run["upper"])
            if run.get("var") is not None:
                sigma = np.sqrt(np.maximum(np.asarray(run["var"]), 0.0))
                ax_state.fill_between(
                    time, lower - sigma, upper + sigma, alpha=0.08, color=color
                )
            ax_state.fill_between(time, lower, upper, alpha=0.18, color=color)
            ax_state.plot(
                time, (lower + upper) / 2, color=color, lw=1.8, marker="o", ms=3,
                label=name,
            )
        else:
            mean = np.asarray(run["mean"])
            sigma = np.sqrt(np.maximum(np.asarray(run["var"]), 0.0))
            ax_state.fill_between(time, mean - sigma, mean + sigma, alpha=0.18, color=color)
            ax_state.plot(time, mean, color=color, lw=1.8, marker="o", ms=3, label=name)

        bounds = _to_numpy(run["formula"])
        t = time[: len(bounds)]
        ax_form.plot(t, bounds[:, 0], color=color, lw=1.8, marker="o", ms=5, label=f"{name} lower")
        if not np.allclose(bounds[:, 0], bounds[:, 1], atol=_COINCIDENT):
            ax_form.plot(t, bounds[:, 1], color=color, lw=1.2, ls="--", marker="o",
                         ms=4, label=f"{name} upper")

    for th in np.atleast_1d(thresholds) if thresholds is not None else []:
        ax_state.axhline(float(th), color=_RED, ls="--", lw=1.3)

    ax_state.set_xlabel("time [s]")
    ax_state.set_ylabel("state $x$")
    ax_state.set_title("(a) State (band = descriptor bounds or $\\mu \\pm \\sigma$)",
                       loc="left", fontweight="bold")
    ax_state.legend(fontsize=8)
    ax_state.grid(True, alpha=0.3)

    ax_form.set_xlabel("time [s]")
    ax_form.set_ylabel("stochastic robustness")
    ax_form.set_ylim(-0.05, 1.05)
    # Keep the full horizon on the axis; a single valid origin would otherwise
    # autoscale to a meaningless window around one point.
    ax_form.set_xlim(time[0] - 0.05, time[-1] + 0.05)
    ax_form.set_title("(b) Formula interval (direct)", loc="left", fontweight="bold")
    ax_form.legend(fontsize=8)
    ax_form.grid(True, alpha=0.3)

    ax_obj.plot(objective_history, color=_GREEN, lw=1.5)
    ax_obj.set_xlabel("iteration")
    ax_obj.set_ylabel("objective $J$")
    ax_obj.set_title("(c) Optimisation objective", loc="left", fontweight="bold")
    ax_obj.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    _finish(fig, save_path, show)
    return fig, (ax_state, ax_form, ax_obj)
