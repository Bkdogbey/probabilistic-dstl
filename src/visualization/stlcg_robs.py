import matplotlib.pyplot as plt
import numpy as np


def plot_predicate_robustness(
    time, mean_trace, var_trace, robustness, formula, threshold
):
    """
    Plot belief trajectory and its robustness trace together.

    Parameters
    ----------
    time : array_like
        Array of time values.
    mean_trace : array_like
        Mean trajectory over time.
    var_trace : array_like
        Variance trajectory over time.
    robustness : array_like
        Robustness values at each timestep (probability in [0, 1]).
    formula : STL_Formula
        The STL formula object (for title).
    threshold : float
        Threshold value for the predicate.
    """
    sigma = np.sqrt(var_trace)
    lower_sigma = mean_trace - sigma
    upper_sigma = mean_trace + sigma

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # TOP PLOT: TRAJECTORY WITH UNCERTAINTY

    # Main trajectory
    ax1.plot(time, mean_trace, color="blue", linewidth=2, label="Mean Height")
    ax1.fill_between(
        time,
        lower_sigma,
        upper_sigma,
        color="blue",
        alpha=0.2,
        label="±1σ Interval",
    )

    # Threshold line
    ax1.axhline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold} m",
    )

    # Identify violation regions
    full_violation = upper_sigma < threshold
    partial_violation = (lower_sigma < threshold) & ~full_violation

    # Vertical range for shading
    y_min = np.min(lower_sigma) * 0.9
    y_max = np.max(upper_sigma) * 1.1

    # Shaded violation regions
    ax1.fill_between(
        time,
        y_min,
        y_max,
        where=full_violation,
        color="red",
        alpha=0.1,
        label="Definitely below threshold",
    )
    ax1.fill_between(
        time,
        y_min,
        y_max,
        where=partial_violation,
        color="orange",
        alpha=0.1,
        label="Uncertainty crosses threshold",
    )

    # Aesthetics
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Height (m)", fontsize=12)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Belief Trajectory with Uncertainty", fontsize=13, fontweight="bold")

    # STOCHASTIC ROBUSTNESS

    # Main robustness curve
    ax2.plot(time, robustness, "b-", linewidth=2.5, label="Stochastic Robustness")

    # Reference lines
    ax2.axhline(
        0.5,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="50% Confidence",
        alpha=0.7,
    )
    ax2.axhline(
        0.9,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="90% Confidence",
        alpha=0.7,
    )

    # Aesthetics
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Robustness\n(Probability)", fontsize=12)
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    title = f"Stochastic Robustness: {formula}"
    ax2.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.show()
