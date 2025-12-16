

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
    show_upper=True,  # Set to False to show only lower bound
):
    """
    Parameters
    ----------
    time : array_like
        Time vector
    robustness_trace : torch.Tensor
        Output from formula(belief_trajectory) with shape [B, T, 2]
    mean_trace : array_like, optional
        Mean signal trajectory
    var_trace : array_like, optional
        Variance of signal trajectory
    thresholds : float or list of float, optional
        Threshold(s) to display on signal plot
        - Single value: thresholds=50.0
        - Multiple: thresholds=[45.0, 55.0]
    formula_str : str, optional
        String representation of the STL formula
    show_upper : bool, optional
        If True: plot both P_lower and P_upper (default)
        If False: plot only P_lower (conservative estimate)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Handle tensor conversion
    if isinstance(robustness_trace, torch.Tensor):
        robustness_trace = robustness_trace.detach().cpu().numpy()

    # Handle dimensions
    if robustness_trace.ndim == 3:
        if robustness_trace.shape[0] == 1:
            robustness_trace = robustness_trace.squeeze(0)
        else:
            robustness_trace = robustness_trace[0]

    # Extract bounds
    prob_lower = robustness_trace[:, 0]
    prob_upper = robustness_trace[:, 1]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # =========================================================================
    # TOP PANEL: Signal Trajectory
    # =========================================================================
    if mean_trace is not None and var_trace is not None:
        mean_trace = np.asarray(mean_trace)
        var_trace = np.asarray(var_trace)
        sigma = np.sqrt(var_trace)
        lower_sigma = mean_trace - sigma
        upper_sigma = mean_trace + sigma

        # Plot signal
        ax1.plot(time, mean_trace, color="navy", linewidth=2.5, label="Mean μ(t)")
        ax1.fill_between(
            time,
            lower_sigma,
            upper_sigma,
            color="steelblue",
            alpha=0.3,
            label="μ(t) ± σ(t)",
        )

        # Add threshold(s)
        if thresholds is not None:
            # Convert single threshold to list
            if not isinstance(thresholds, (list, tuple)):
                thresholds = [thresholds]

            # Plot each threshold
            colors = ["red", "orange", "purple", "brown"]
            for i, thresh in enumerate(thresholds):
                color = colors[i % len(colors)]
                ax1.axhline(
                    thresh,
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    label=f"Threshold: x ≥ {thresh}",
                )

        ax1.set_ylabel("Signal x(t)", fontsize=12, fontweight="bold")
        ax1.set_title("Stochastic Signal Trajectory", fontsize=13, fontweight="bold")
        ax1.legend(loc="best", fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

    # =========================================================================
    # BOTTOM PANEL: STL Satisfaction Probability
    # =========================================================================

    if show_upper:
        # Plot both bounds
        ax2.plot(
            time,
            prob_lower,
            linewidth=2.5,
            color="darkgreen",
            label="P_lower (worst-case)",
            marker="o",
            markersize=3,
            markevery=max(1, len(time) // 30),
        )

        ax2.plot(
            time,
            prob_upper,
            linewidth=2.5,
            color="limegreen",
            label="P_upper (best-case)",
            marker="s",
            markersize=3,
            markevery=max(1, len(time) // 30),
        )

        # Fill interval between bounds
        ax2.fill_between(
            time,
            prob_lower,
            prob_upper,
            alpha=0.25,
            color="green",
            label="Uncertainty interval",
        )

        # Interpretation note
        textstr = (
            "Interpretation:\n"
            "P_lower: Guaranteed probability\n"
            "P_upper: Best-case probability\n"
            "Green band: Uncertainty"
        )
    else:
        # Plot only lower bound
        ax2.plot(
            time,
            prob_lower,
            linewidth=2.5,
            color="darkgreen",
            label="P(φ satisfied)",
            marker="o",
            markersize=3,
            markevery=max(1, len(time) // 30),
        )

        # Fill under curve
        ax2.fill_between(time, 0, prob_lower, alpha=0.25, color="green")

        # Interpretation note
        textstr = (
            "Interpretation:\nHigh P → Formula satisfied\nLow P → Formula violated"
        )

    # Add reference lines
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.6)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.4)

    # Add reference labels
    ax2.text(time[-1] * 1.01, 1.0, "P = 1", fontsize=9, va="center", alpha=0.6)
    ax2.text(time[-1] * 1.01, 0.5, "P = 0.5", fontsize=9, va="center", alpha=0.6)
    ax2.text(time[-1] * 1.01, 0.0, "P = 0", fontsize=9, va="center", alpha=0.6)

    # Labels and title
    ax2.set_xlabel("Time t (s)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Satisfaction Probability", fontsize=12, fontweight="bold")
    ax2.set_ylim([-0.05, 1.05])

    # Title with formula
    if formula_str:
        title = f"STL Formula: {formula_str}"
    else:
        title = "STL Satisfaction Probability"
    ax2.set_title(title, fontsize=13, fontweight="bold")

    ax2.legend(loc="best", fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Add interpretation box
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    ax2.text(
        0.02,
        0.98,
        textstr,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.show()

    return fig
