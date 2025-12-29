import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_stl_formula_bounds(
    time,
    robustness_trace,
    mean_trace=None,
    var_trace=None,
    thresholds=None,
    formula_str=None,
    show_upper=True,
    interval=None,  # [a,b] in STEPS (not seconds)
    show_windows=True,
    n_example_windows=3,
):
    """
    General STL verification plot with optional temporal-window visualization.

    Parameters
    ----------
    time : array_like
        Time vector (seconds or arbitrary units), length T
    robustness_trace : torch.Tensor or np.ndarray
        Output from formula(belief_trajectory), shape [B, T, 2] or [T,2]
    mean_trace : array_like, optional
        Mean signal trajectory, length T
    var_trace : array_like, optional
        Variance trajectory, length T
    thresholds : float or list[float], optional
        Threshold line(s) to display on signal plot
    formula_str : str, optional
        String representation of the STL formula
    show_upper : bool
        Plot both lower/upper bounds if True, else only lower
    interval : (a,b), optional
        Temporal interval [a,b] IN STEPS (indices). Used only for visualization.
    show_windows : bool
        If True and interval provided, visualize windows [t+a, t+b]
    n_example_windows : int
        Number of example windows to visualize

    Returns
    -------
    fig, (ax1, ax2)
    """
    time = np.asarray(time)
    T = len(time)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _pick_example_indices(n_points: int, n_examples: int) -> np.ndarray:
        if n_points <= 1:
            return np.array([0], dtype=int)
        idx = np.linspace(0, n_points - 1, num=min(n_examples, n_points))
        return np.unique(np.round(idx).astype(int))

    def _as_np_trace(rt):
        if isinstance(rt, torch.Tensor):
            rt = rt.detach().cpu().numpy()
        rt = np.asarray(rt)

        # Accept [B,T,2], [T,2], [B,T] (rare), [T]
        if rt.ndim == 3:
            rt = rt[0]  # take batch 0 -> [T,2]
        elif rt.ndim == 2:
            # [T,2] ok
            pass
        elif rt.ndim == 1:
            # [T] -> treat as lower=upper
            rt = np.stack([rt, rt], axis=-1)
        else:
            raise ValueError(f"Unexpected robustness_trace shape: {rt.shape}")

        if rt.shape[0] != T:
            raise ValueError(
                f"Time length T={T} but robustness_trace has length {rt.shape[0]}."
            )
        if rt.shape[1] != 2:
            raise ValueError(
                f"robustness_trace must have last dim 2 (lower/upper). Got {rt.shape}."
            )
        return rt

    rt = _as_np_trace(robustness_trace)
    prob_lower = rt[:, 0]
    prob_upper = rt[:, 1]

    # Robust dt for inclusive-looking shading
    dt = np.median(np.diff(time)) if T > 1 else 1.0

    # -----------------------------
    # Figure
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # =========================================================================
    # TOP PANEL: Signal
    # =========================================================================
    if mean_trace is not None and var_trace is not None:
        mean_trace = np.asarray(mean_trace)
        var_trace = np.asarray(var_trace)

        if len(mean_trace) != T or len(var_trace) != T:
            raise ValueError("mean_trace and var_trace must have same length as time.")

        sigma = np.sqrt(np.maximum(var_trace, 0.0))
        lower_sigma = mean_trace - sigma
        upper_sigma = mean_trace + sigma

        ax1.plot(time, mean_trace, linewidth=2.5, label="Mean μ(t)", zorder=5)
        ax1.fill_between(
            time,
            lower_sigma,
            upper_sigma,
            alpha=0.25,
            label="μ(t) ± σ(t)",
            zorder=3,
        )

        # Threshold(s)
        if thresholds is not None:
            if not isinstance(thresholds, (list, tuple)):
                thresholds = [thresholds]
            for thresh in thresholds:
                ax1.axhline(
                    thresh,
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.8,
                    label=f"Threshold = {thresh}",
                    zorder=4,
                )

        ax1.set_ylabel("Signal x(t)", fontsize=13, fontweight="bold")
        ax1.set_title("Signal Trajectory", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, zorder=0)

        # Add interval annotation
        if interval is not None:
            a, b = int(interval[0]), int(interval[1])
            ax1.text(
                0.01,
                0.98,
                f"Interval (steps): [{a},{b}]",
                transform=ax1.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.12),
            )

    else:
        # Still label the axis
        ax1.set_ylabel("Signal x(t)", fontsize=13, fontweight="bold")
        ax1.set_title("Signal Trajectory", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, zorder=0)

    # After signal is drawn, get limits for arrow placement
    ymin, ymax = ax1.get_ylim()
    y_arrow = ymin + 0.85 * (ymax - ymin)

    # =========================================================================
    # Window visualization on TOP
    # =========================================================================
    if show_windows and interval is not None and T > 0:
        a, b = int(interval[0]), int(interval[1])

        example_indices = _pick_example_indices(T, n_example_windows)
        colors_window = ["orange", "purple", "brown", "cyan", "magenta"]

        for i, t_idx in enumerate(example_indices):
            color = colors_window[i % len(colors_window)]

            # window indices in step-space; clip only for visualization
            ws = int(np.clip(t_idx + a, 0, T - 1))
            we = int(np.clip(t_idx + b, 0, T - 1))

            t_current = time[t_idx]
            t_start = time[ws]
            t_end = time[we]

            # shaded region (inclusive-looking)
            ax1.axvspan(
                t_start,
                t_end + 0.5 * dt,
                alpha=0.12,
                color=color,
                zorder=1,
            )

            # line at current time
            ax1.axvline(
                t_current,
                color=color,
                linestyle=":",
                linewidth=2.0,
                alpha=0.65,
                zorder=4,
            )

            # arrow from t to start of window
            ax1.annotate(
                "",
                xy=(t_start, y_arrow),
                xytext=(t_current, y_arrow),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0, alpha=0.7),
                zorder=6,
            )

            ax1.text(
                t_current,
                y_arrow,
                f" t={t_current:.2f}",
                fontsize=9,
                color=color,
                fontweight="bold",
                va="bottom",
                ha="left",
                zorder=7,
            )

    # Legend for top (after windows so labels don’t get duplicated)
    ax1.legend(loc="best", fontsize=10, framealpha=0.95, ncol=2)

    # =========================================================================
    # BOTTOM PANEL: Probability bounds
    # =========================================================================
    if show_upper:
        ax2.plot(
            time,
            prob_lower,
            linewidth=2.5,
            label="P_lower (guaranteed)",
            marker="o",
            markersize=4,
            markevery=max(1, T // 25),
            zorder=5,
        )
        ax2.plot(
            time,
            prob_upper,
            linewidth=2.5,
            label="P_upper (best-case)",
            marker="s",
            markersize=4,
            markevery=max(1, T // 25),
            zorder=5,
        )
        ax2.fill_between(
            time,
            prob_lower,
            prob_upper,
            alpha=0.25,
            label="Uncertainty interval",
            zorder=3,
        )
    else:
        ax2.plot(
            time,
            prob_lower,
            linewidth=3.0,
            label="P(φ satisfied)",
            marker="o",
            markersize=4,
            markevery=max(1, T // 25),
            zorder=5,
        )
        ax2.fill_between(time, 0, prob_lower, alpha=0.25, zorder=3)

    # Window visualization on BOTTOM
    if show_windows and interval is not None and T > 0:
        a, b = int(interval[0]), int(interval[1])
        example_indices = _pick_example_indices(T, n_example_windows)
        colors_window = ["orange", "purple", "brown", "cyan", "magenta"]

        for i, t_idx in enumerate(example_indices):
            color = colors_window[i % len(colors_window)]
            ws = int(np.clip(t_idx + a, 0, T - 1))
            we = int(np.clip(t_idx + b, 0, T - 1))

            ax2.axvspan(
                time[ws],
                time[we] + 0.5 * dt,
                alpha=0.08,
                color=color,
                zorder=1,
            )
            ax2.axvline(
                time[t_idx],
                color=color,
                linestyle=":",
                linewidth=1.8,
                alpha=0.5,
                zorder=4,
            )

    # Reference lines
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.35, zorder=2)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.55, zorder=2)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.35, zorder=2)

    ax2.set_xlabel("Time t (s)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Satisfaction Probability", fontsize=13, fontweight="bold")
    ax2.set_ylim([-0.05, 1.05])

    if formula_str:
        ax2.set_title(f"STL Formula: {formula_str}", fontsize=14, fontweight="bold")
    else:
        ax2.set_title("STL Satisfaction Probability", fontsize=14, fontweight="bold")

    ax2.legend(loc="best", fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.show()
    return fig, (ax1, ax2)
