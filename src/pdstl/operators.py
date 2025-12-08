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
    Generic visualization for ANY STL formula in operators.py.

    time: array-like [T]
    robustness_trace: formula(belief_trajectory), usually [B,T,D,2] or [T,2] or [2]
    """

    time = np.asarray(time)
    T = time.shape[0]

    # ----------------------------------------------------------------------
    # Convert to numpy
    # ----------------------------------------------------------------------
    if isinstance(robustness_trace, torch.Tensor):
        rt = robustness_trace.detach().cpu().numpy()
    else:
        rt = np.asarray(robustness_trace)

    # ----------------------------------------------------------------------
    # Try to interpret shape as [T,2]
    # ----------------------------------------------------------------------
    if rt.ndim == 4:
        # assume [B,T,D,2]
        rt = rt[0, :, 0, :]  # [T,2]
    elif rt.ndim == 3:
        # could be [T,D,2] or [B,T,2] with singleton dims
        if rt.shape[-1] == 2 and (rt.shape[0] == T):
            rt = rt[:, 0, :] if rt.shape[1] == 1 else rt[0, :, :]
        else:
            rt = rt.reshape(-1, 2)
    elif rt.ndim == 2 and rt.shape[1] == 2:
        # already [*,2]
        pass
    elif rt.ndim == 1:
        # Could be just [2] = [lower, upper] or [1] = single prob
        if rt.shape[0] == 2:
            rt = rt.reshape(1, 2)  # [1,2]
        else:
            rt = np.stack([rt, rt], axis=-1)  # [1,2]
    else:
        raise ValueError(f"Unexpected robustness_trace shape: {rt.shape}")

    # ----------------------------------------------------------------------
    # Now rt is [N,2]. We want length T.
    # If N == 1, treat it as a single global value and broadcast.
    # ----------------------------------------------------------------------
    if rt.shape[0] == 1 and T > 1:
        rt = np.repeat(rt, T, axis=0)  # [T,2]
    elif rt.shape[0] != T:
        raise ValueError(
            f"Time length {T} and robustness length {rt.shape[0]} differ "
            f"(cannot align automatically)"
        )

    lower = rt[:, 0]
    upper = rt[:, 1]

    
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

        # ----------------- TOP: signal + uncertainty ----------------------
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

    # ----------------- BOTTOM: [lower, upper] band -----------------------
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

    for level in (0.5, 0.9):
        ax.axhline(
            level,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"{int(level * 100)}% level",
        )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    global_lower = float(np.min(lower))
    ax.set_title(
        f"STL formula {formula}\n"
        f"Temporal probability bounds (worst-case lower = {global_lower:.3f})",
        fontsize=12,
        fontweight="bold",
    )

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=9, loc="best")

    plt.tight_layout()
    plt.show()
