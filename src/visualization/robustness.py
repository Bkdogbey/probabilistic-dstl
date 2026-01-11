"""
STL Robustness Visualization

3-panel layout:
  (a) Signal trajectory
  (b) Predicate satisfaction probability
  (c) Temporal operator output
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(trace, T):
    """Convert trace to numpy [T, 2]."""
    if isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()
    trace = np.asarray(trace)

    if trace.ndim == 3:
        trace = trace[0]
    elif trace.ndim == 1:
        trace = np.stack([trace, trace], axis=-1)

    assert trace.shape == (T, 2), f"Expected ({T}, 2), got {trace.shape}"
    return trace


def plot_stl_formula_bounds(
    time,
    robustness_trace,
    mean_trace=None,
    var_trace=None,
    predicate_trace=None,
    thresholds=None,
    formula_str=None,
    interval=None,
    operator_type="always",
    figsize=(10, 8),
    save_path=None,
):
    """
    STL verification plot with signal, predicate, and operator output.

    Parameters
    ----------
    time : array_like
        Time vector, length T
    robustness_trace : array_like
        Output of temporal operator, shape [B,T,2] or [T,2]
    mean_trace, var_trace : array_like, optional
        Signal mean and variance for top panel
    predicate_trace : array_like, optional
        Output of predicate φ, shape [B,T,2] or [T,2]
    thresholds : float or list, optional
        Threshold line(s) for signal plot
    formula_str : str, optional
        Formula string for suptitle
    interval : [a, b], optional
        Temporal interval in STEPS (auto windows if provided)
    operator_type : str
        'always' or 'eventually'
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig, axes
    """
    time = np.asarray(time)
    T = len(time)

    oper = _to_numpy(robustness_trace, T)
    pred = _to_numpy(predicate_trace, T) if predicate_trace is not None else None

    # Operator symbol
    op_symbol = "□" if operator_type == "always" else "◇"

    # Figure setup
    if pred is not None:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        ax_signal, ax_pred, ax_oper = axes
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 0.7), sharex=True
        )
        ax_signal, ax_oper = axes
        ax_pred = None

    # =========================================================================
    # Panel (a): Signal
    # =========================================================================
    if mean_trace is not None and var_trace is not None:
        mean_trace = np.asarray(mean_trace)
        var_trace = np.asarray(var_trace)
        sigma = np.sqrt(np.maximum(var_trace, 0.0))

        ax_signal.fill_between(
            time, mean_trace - sigma, mean_trace + sigma, alpha=0.25, color="steelblue"
        )
        ax_signal.plot(time, mean_trace, "b-", lw=1.5, label="μ(t)")
        ax_signal.plot(time, mean_trace + sigma, "b--", lw=1, alpha=0.7, label="μ ± σ")
        ax_signal.plot(time, mean_trace - sigma, "b--", lw=1, alpha=0.7)

        if thresholds is not None:
            thresholds = (
                [thresholds]
                if not isinstance(thresholds, (list, tuple))
                else thresholds
            )
            for th in thresholds:
                ax_signal.axhline(th, color="red", ls="--", lw=1.5, label=f"h = {th}")

    ax_signal.set_ylabel("x(t)")
    ax_signal.set_title("(a) Signal Trajectory", loc="left", fontweight="bold")
    ax_signal.legend(loc="upper right", fontsize=8)
    ax_signal.grid(True, alpha=0.3)

    # =========================================================================
    # Panel (b): Predicate with sliding windows
    # =========================================================================
    if ax_pred is not None:
        # Light green fill between bounds
        ax_pred.fill_between(
            time, pred[:, 0], pred[:, 1], alpha=0.3, color="lightgreen"
        )
        ax_pred.plot(time, pred[:, 0], "b-", lw=1.5, label="Lower bound")
        ax_pred.plot(time, pred[:, 1], "r-", lw=1.5, label="Upper bound")

        # Auto sliding windows
        if interval is not None:
            _draw_sliding_windows(ax_pred, time, T, interval)

        ax_pred.set_ylabel("P(φ)")
        ax_pred.set_ylim(-0.05, 1.05)
        ax_pred.set_title(
            "(b) Predicate Satisfaction Probability", loc="left", fontweight="bold"
        )
        ax_pred.legend(loc="upper right", fontsize=8)
        ax_pred.grid(True, alpha=0.3)

    # =========================================================================
    # Panel (c): Operator output
    # =========================================================================
    # Light green fill between bounds
    ax_oper.fill_between(time, oper[:, 0], oper[:, 1], alpha=0.3, color="lightgreen")
    ax_oper.plot(time, oper[:, 0], "b-", lw=1.5, label="Lower bound")
    ax_oper.plot(time, oper[:, 1], "r-", lw=1.5, label="Upper bound")

    # Auto sliding windows with output markers
    if interval is not None and pred is not None:
        _draw_output_markers(ax_oper, time, T, interval, oper)

    ax_oper.set_xlabel("Time (s)")
    ax_oper.set_ylabel(f"P({op_symbol}φ)")
    ax_oper.set_ylim(-0.05, 1.05)
    ax_oper.set_title(f"(c) Temporal Operator Output", loc="left", fontweight="bold")
    ax_oper.legend(loc="upper right", fontsize=8)
    ax_oper.grid(True, alpha=0.3)

    # Suptitle with formula
    if formula_str:
        fig.suptitle(formula_str, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    return fig, axes


def _draw_sliding_windows(ax, time, T, interval):
    """Draw automatic sliding windows based on interval."""
    a = int(interval[0])
    b_inf = np.isinf(interval[1])
    b = T - 1 if b_inf else int(interval[1])

    # Window width in steps
    window_width = b - a + 1

    # Auto-select number of windows: show ~4 non-overlapping if possible
    if window_width >= T:
        # Window covers entire trace, just show one
        n_windows = 1
    else:
        # Aim for 4 windows, spaced so they don't overlap too much
        n_windows = min(4, max(1, T // window_width))

    # Pick evenly spaced starting points
    max_start = max(0, T - 1 - b)
    if max_start <= 0:
        t_indices = np.array([0])
    else:
        t_indices = np.linspace(0, max_start, n_windows).astype(int)

    # Colors - varying orange intensity
    colors = plt.cm.Oranges(np.linspace(0.25, 0.65, len(t_indices)))

    for i, t_idx in enumerate(t_indices):
        w_start = t_idx + a
        w_end = t_idx + b

        # Clip to valid range
        w_start = min(w_start, T - 1)
        w_end = min(w_end, T - 1)

        if w_start > w_end:
            continue

        # Shaded window
        ax.axvspan(time[w_start], time[w_end], alpha=0.15, color=colors[i])

        # Small marker at evaluation point t_idx
        ax.plot(
            time[t_idx], -0.03, marker="v", color=colors[i], markersize=5, clip_on=False
        )


def _draw_output_markers(ax, time, T, interval, oper):
    """Draw markers on output corresponding to window positions."""
    a = int(interval[0])
    b_inf = np.isinf(interval[1])
    b = T - 1 if b_inf else int(interval[1])

    window_width = b - a + 1

    if window_width >= T:
        n_windows = 1
    else:
        n_windows = min(4, max(1, T // window_width))

    max_start = max(0, T - 1 - b)
    if max_start <= 0:
        t_indices = np.array([0])
    else:
        t_indices = np.linspace(0, max_start, n_windows).astype(int)

    colors = plt.cm.Oranges(np.linspace(0.25, 0.65, len(t_indices)))

    for i, t_idx in enumerate(t_indices):
        if t_idx >= T:
            continue
        ax.plot(time[t_idx], oper[t_idx, 0], "o", color=colors[i], markersize=5)
