import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_stl_formula_bounds(
    time,
    robustness_trace,
    formula,
    mean_trace=None,
    var_trace=None,
    signal_label="x(t)",
):
    """
    Minimal, generic visualization for ANY STL formula.

    - Accepts output of formula(belief_trajectory).
    - Extracts lower & upper probability.
    - If only a single value is returned, repeats it across time.
    """

    time = np.asarray(time)
    T = time.shape[0]

    # ------------------------------------------------------------------
    # 1. Convert robustness_trace to numpy
    # ------------------------------------------------------------------
    if isinstance(robustness_trace, torch.Tensor):
        rt = robustness_trace.detach().cpu().numpy()
    else:
        rt = np.asarray(robustness_trace)

    # rt could be:
    #   [B,T,D,2], [T,2], [2], [1,2], [1,1,1,2], etc.
    # We just want [T,2] in the end.
    # ------------------------------------------------------------------
    if rt.ndim == 4:
        # assume [B,T,D,2]
        rt = rt[0, :, 0, :]  # [T,2] or [1,2] depending on your operator
    elif rt.ndim == 3:
        # try to collapse any singleton dims
        rt = rt.squeeze()
    elif rt.ndim == 1:
        # could be [2] or [1]
        if rt.shape[0] == 2:
            rt = rt.reshape(1, 2)  # [1,2]
        else:
            rt = np.stack([rt, rt], axis=-1)  # [1,2]

    # At this point rt should be [N,2]
    if rt.ndim == 1:
        # just in case
        rt = rt.reshape(1, 2)

    if rt.shape[-1] != 2:
        raise ValueError(f"robustness_trace last dim must be 2, got shape {rt.shape}")

    N = rt.shape[0]

    # ------------------------------------------------------------------
    # 2. If we only have 1 time value, broadcast it across all T
    # ------------------------------------------------------------------
    if N == 1 and T > 1:
        rt = np.repeat(rt, T, axis=0)  # [T,2]
    elif N != T:
        # Here we refuse to guess further
        raise ValueError(
            f"Time length {T} and robustness length {N} differ; "
            "your operator is not returning per-time robustness."
        )

    lower = rt[:, 0]
    upper = rt[:, 1]

    # Debug print so you SEE the numbers in the terminal
    print(f"[DEBUG] Formula: {formula}")
    print(f"[DEBUG] Lower bounds: min={lower.min():.4f}, max={lower.max():.4f}")
    print(f"[DEBUG] Upper bounds: min={upper.min():.4f}, max={upper.max():.4f}")

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------
    has_signal = mean_trace is not None and var_trace is not None

    if has_signal:
        mean_trace = np.asarray(mean_trace)
        var_trace = np.asarray(var_trace)
        sigma = np.sqrt(var_trace)
        lower_sigma = mean_trace - sigma
        upper_sigma = mean_trace + sigma

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, height_ratios=[2, 1]
        )

        # Signal + uncertainty
        ax1.plot(time, mean_trace, linewidth=2, label="Mean trajectory")
        ax1.fill_between(
            time,
            lower_sigma,
            upper_sigma,
            alpha=0.2,
            label="±1σ uncertainty",
        )
        ax1.set_ylabel(signal_label, fontsize=11)
        ax1.set_title(
            "Signal and uncertainty (input to STL spec)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        ax = ax2
    else:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Probability interval
    ax.fill_between(
        time,
        lower,
        upper,
        alpha=0.2,
        label="Probability interval [lower, upper]",
    )
    ax.plot(time, lower, linewidth=2, label="Lower bound l(t)")
    if not np.allclose(lower, upper):
        ax.plot(time, upper, linewidth=2, linestyle="--", label="Upper bound u(t)")

    # Reference levels
    for level in (0.5, 0.9):
        ax.axhline(
            level,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"{int(level*100)}% level",
        )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    global_lower = float(np.min(lower))
    ax.set_title(
        f"STL formula {formula}\n"
        f"Worst-case lower bound = {global_lower:.3f}",
        fontsize=12,
        fontweight="bold",
    )

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=9, loc="best")

    plt.tight_layout()
    plt.show()
