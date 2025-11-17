import matplotlib.pyplot as plt
import numpy as np


def plot_predicate_robustness(
    time, mean_trace, var_trace, robustness, formula, threshold
):
    """
    Plot belief trajectory, pointwise robustness, and stochastic robustness together.

    Parameters
    ----------
    time : array_like
        Array of time values.
    mean_trace : array_like
        Mean trajectory over time.
    var_trace : array_like
        Variance trajectory over time.
    robustness : array_like
        Stochastic robustness values (probabilities in [0, 1]).
    formula : STL_Formula
        The STL formula object.
    threshold : float
        Threshold value for the predicate.
    """
    sigma = np.sqrt(var_trace)
    lower_sigma = mean_trace - sigma
    upper_sigma = mean_trace + sigma

    pointwise_robustness = lower_sigma - threshold

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[2, 1, 1])

    # TOP PLOT: TRAJECTORY WITH UNCERTAINTY

    # Main trajectory
    ax1.plot(time, mean_trace, color="blue", linewidth=2, label="Mean Height")
    ax1.fill_between(
        time,
        lower_sigma,
        upper_sigma,
        color="blue",
        alpha=0.2,
        label="±1sigma Uncertainty",
    )

    # Threshold line
    ax1.axhline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold}",
    )

    # Aesthetics
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Height", fontsize=11)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Belief Trajectory", fontsize=12, fontweight="bold")

    # POINTWISE ROBUSTNESS (Deterministic)

    ax2.plot(
        time,
        pointwise_robustness,
        "b-",
        linewidth=2,
        label="Pointwise Robustness (lower bound - threshold)",
    )

    # Zero line (satisfaction boundary)
    ax2.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=2,
        label="Satisfaction Boundary (ρ=0)",
    )

    # Fill satisfaction/violation regions
    ax2.fill_between(
        time,
        0,
        pointwise_robustness,
        where=(pointwise_robustness >= 0),
        color="green",
        alpha=0.2,
        label="Satisfied (ρ>0)",
        interpolate=True,
    )
    ax2.fill_between(
        time,
        pointwise_robustness,
        0,
        where=(pointwise_robustness < 0),
        color="red",
        alpha=0.2,
        label="Violated (ρ<0)",
        interpolate=True,
    )

    # Aesthetics
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Pointwise Robustness", fontsize=11)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(
        "Pointwise Robustness (Deterministic)", fontsize=12, fontweight="bold"
    )

    # STOCHASTIC ROBUSTNESS

    ax3.plot(time, robustness, "b-", linewidth=2.5, label="Stochastic Robustness")

    # Reference lines
    ax3.axhline(
        0.5,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="50% Probability",
        alpha=0.7,
    )
    ax3.axhline(
        0.9,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="90% Confidence",
        alpha=0.7,
    )

    # Aesthetics
    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_ylabel("Stochastic Robustness\n(Probability)", fontsize=11)
    ax3.set_ylim([0, 1.05])
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3)

    min_rob = np.min(robustness)
    ax3.set_title(
        f"Stochastic Robustness: {formula} | Min: {min_rob:.4f}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()
