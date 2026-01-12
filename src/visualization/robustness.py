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
    time = np.asarray(time)
    T = len(time)

    oper = _to_numpy(robustness_trace, T)
    pred = _to_numpy(predicate_trace, T) if predicate_trace is not None else None

    op_symbol = "□" if operator_type == "always" else "◇"

    if pred is not None:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        ax_signal, ax_pred, ax_oper = axes
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 0.7), sharex=True
        )
        ax_signal, ax_oper = axes
        ax_pred = None

    # Panel (a): Signal

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

    # Panel (b): Predicate with sliding windows

    if ax_pred is not None:
        ax_pred.fill_between(
            time, pred[:, 0], pred[:, 1], alpha=0.3, color="lightgreen"
        )
        ax_pred.plot(time, pred[:, 0], "b-", lw=1.5, label="Lower bound")
        ax_pred.plot(time, pred[:, 1], "r-", lw=1.5, label="Upper bound")

        ax_pred.set_ylabel("P(φ)")
        ax_pred.set_ylim(-0.05, 1.05)
        ax_pred.set_title(
            "(b) Predicate Satisfaction Probability", loc="left", fontweight="bold"
        )
        ax_pred.legend(loc="upper right", fontsize=8)
        ax_pred.grid(True, alpha=0.3)

    # Panel (c): Operator output
    ax_oper.fill_between(time, oper[:, 0], oper[:, 1], alpha=0.3, color="lightgreen")
    ax_oper.plot(time, oper[:, 0], "b-", lw=1.5, label="Lower bound")
    ax_oper.plot(time, oper[:, 1], "r-", lw=1.5, label="Upper bound")

    ax_oper.set_xlabel("Time (s)")
    ax_oper.set_ylabel(f"P({op_symbol}φ)")
    ax_oper.set_ylim(-0.05, 1.05)
    ax_oper.set_title(f"(c) Temporal Operator Output", loc="left", fontweight="bold")
    ax_oper.legend(loc="upper right", fontsize=8)
    ax_oper.grid(True, alpha=0.3)

    if formula_str:
        fig.suptitle(formula_str, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig, axes


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


def plot_piecewise_stl(
    time,
    robustness_trace,
    mean_trace=None,
    var_trace=None,
    predicate_trace=None,
    thresholds=None,
    formula_str=None,
    interval=None,
    operator_type="always",
    figsize=(10, 9),
    save_path=None,
):
    time = np.asarray(time)
    T = len(time)

    oper = _to_numpy(robustness_trace, T)
    pred = _to_numpy(predicate_trace, T) if predicate_trace is not None else None

    mean_trace = np.asarray(mean_trace) if mean_trace is not None else None
    var_trace = np.asarray(var_trace) if var_trace is not None else None
    sigma_trace = np.sqrt(var_trace) if var_trace is not None else None

    threshold = (
        thresholds
        if isinstance(thresholds, (int, float))
        else (thresholds[0] if thresholds else 50)
    )

    a, b = interval if interval else [0, 1]
    op_symbol = "□" if operator_type == "always" else "◇"

    # Colors
    signal_color = "black"
    bound_color = "steelblue"
    threshold_color = "red"
    lower_color = "blue"
    upper_color = "red"

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # -------------------------------------------------------------------------
    # Panel (a): Signal Trajectory - Step function
    # -------------------------------------------------------------------------
    ax1 = axes[0]

    if mean_trace is not None and sigma_trace is not None:
        for i in range(T):
            x_start = time[i]
            x_end = time[i + 1] if i < T - 1 else time[i] + 0.5

            # Mean line
            ax1.hlines(mean_trace[i], x_start, x_end, colors=signal_color, lw=2)

            # Uncertainty band (μ ± σ)
            upper_band = mean_trace[i] + sigma_trace[i]
            lower_band = mean_trace[i] - sigma_trace[i]
            ax1.hlines(
                upper_band, x_start, x_end, colors=bound_color, lw=1, ls="--", alpha=0.7
            )
            ax1.hlines(
                lower_band, x_start, x_end, colors=bound_color, lw=1, ls="--", alpha=0.7
            )
            ax1.fill_between(
                [x_start, x_end], lower_band, upper_band, alpha=0.15, color=bound_color
            )

            # Vertical transition
            if i < T - 1:
                ax1.vlines(
                    time[i + 1],
                    mean_trace[i],
                    mean_trace[i + 1],
                    colors=signal_color,
                    lw=2,
                )

            # Marker
            ax1.plot(time[i], mean_trace[i], "ko", markersize=6)

        ax1.axhline(threshold, color=threshold_color, ls="--", lw=1.5)

        # Time labels
        for i in range(T):
            offset = 3 if mean_trace[i] < threshold else -12
            ax1.annotate(
                f"t$_{i}$",
                (time[i], mean_trace[i]),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=10,
            )

        ax1.annotate(
            f"h = {int(threshold)}",
            (time[-1] + 0.3, threshold),
            fontsize=10,
            color=threshold_color,
        )

    ax1.set_ylabel("x(t)", fontsize=11)
    ax1.set_title(
        "(a) Signal Trajectory with Uncertainty μ ± σ",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlim(-0.3, time[-1] + 1)
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel (b): Predicate P(x ≥ h) - Step function
    # -------------------------------------------------------------------------
    ax2 = axes[1]

    if pred is not None:
        for i in range(T):
            x_start = time[i]
            x_end = time[i + 1] if i < T - 1 else time[i] + 0.5

            ax2.hlines(pred[i, 0], x_start, x_end, colors=lower_color, lw=2)
            ax2.hlines(pred[i, 1], x_start, x_end, colors=upper_color, lw=2)
            ax2.fill_between(
                [x_start, x_end], pred[i, 0], pred[i, 1], alpha=0.2, color="green"
            )

            if i < T - 1:
                ax2.vlines(
                    time[i + 1],
                    pred[i, 0],
                    pred[i + 1, 0],
                    colors=lower_color,
                    lw=1,
                    alpha=0.5,
                )
                ax2.vlines(
                    time[i + 1],
                    pred[i, 1],
                    pred[i + 1, 1],
                    colors=upper_color,
                    lw=1,
                    alpha=0.5,
                )

            ax2.plot(time[i], pred[i, 0], "bo", markersize=5)
            ax2.plot(time[i], pred[i, 1], "ro", markersize=5)

    ax2.set_ylabel("P(φ)", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(
        f"(b) Predicate Probability P(x ≥ {int(threshold)})",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax2.plot([], [], "b-", lw=2, label="P$_{lower}$")
    ax2.plot([], [], "r-", lw=2, label="P$_{upper}$")
    ax2.legend(loc="right", fontsize=9)
    ax2.set_xlim(-0.3, time[-1] + 1)
    ax2.grid(True, alpha=0.3)

    # Panel (c): Operator  Step function

    ax3 = axes[2]

    for i in range(T):
        x_start = time[i]
        x_end = time[i + 1] if i < T - 1 else time[i] + 0.5

        ax3.hlines(oper[i, 0], x_start, x_end, colors=lower_color, lw=2)
        ax3.hlines(oper[i, 1], x_start, x_end, colors=upper_color, lw=2)
        ax3.fill_between(
            [x_start, x_end], oper[i, 0], oper[i, 1], alpha=0.2, color="green"
        )

        if i < T - 1:
            ax3.vlines(
                time[i + 1],
                oper[i, 0],
                oper[i + 1, 0],
                colors=lower_color,
                lw=1,
                alpha=0.5,
            )
            ax3.vlines(
                time[i + 1],
                oper[i, 1],
                oper[i + 1, 1],
                colors=upper_color,
                lw=1,
                alpha=0.5,
            )

        ax3.plot(time[i], oper[i, 0], "bo", markersize=5)
        ax3.plot(time[i], oper[i, 1], "ro", markersize=5)

    ax3.set_xlabel("Time t", fontsize=11)
    ax3.set_ylabel(f"P({op_symbol}φ)", fontsize=11)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title(
        f"(c) {operator_type.capitalize()} Operator {op_symbol}[{a},{b}](x ≥ {int(threshold)})",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax3.plot([], [], "b-", lw=2, label=f"{op_symbol}P$_{{lower}}$")
    ax3.plot([], [], "r-", lw=2, label=f"{op_symbol}P$_{{upper}}$")
    ax3.legend(loc="right", fontsize=9)
    ax3.set_xlim(-0.3, time[-1] + 1)
    ax3.set_xticks(time)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

    return fig, axes
