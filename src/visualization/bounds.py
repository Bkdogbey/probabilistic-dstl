import matplotlib.pyplot as plt
import numpy as np


def plot_mean_with_sigma_bounds(time, mean_trace, var_trace, threshold=50):
    """
    Plot a trace with ±1σ bounds and highlight where the trace
    violates a given threshold.

    Parameters
    ----------
    time : array_like
        Array of time values.
    mean_trace : array_like
        Mean trace over time.
    var_trace : array_like
        Variance trace over time.
    threshold : float, optional
        Threshold value for violation detection (default is 50).
    """
    sigma = np.sqrt(var_trace)
    lower_sigma = mean_trace - sigma
    upper_sigma = mean_trace + sigma

    # Identify violations
    full_violation = upper_sigma < threshold
    partial_violation = (lower_sigma < threshold) & ~full_violation

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 6))

    # Main trace and uncertainty
    ax.plot(time, mean_trace, color="blue", linewidth=2, label="Mean Height")
    ax.fill_between(
        time,
        lower_sigma,
        upper_sigma,
        color="blue",
        alpha=0.2,
        label="±1σ Interval",
    )

    # Threshold line
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold = {threshold} m",
    )

    # Vertical range for shading violations
    y_min = np.min(lower_sigma) * 0.9
    y_max = np.max(upper_sigma) * 1.1

    # Shaded violation regions
    ax.fill_between(
        time,
        y_min,
        y_max,
        where=full_violation,
        color="red",
        alpha=0.1,
        label="Full violation (both bounds < threshold)",
    )
    ax.fill_between(
        time,
        y_min,
        y_max,
        where=partial_violation,
        color="orange",
        alpha=0.1,
        label="Partial violation (lower bound < threshold)",
    )

    # Violation markers
    ax.scatter(
        time[full_violation],
        mean_trace[full_violation],
        color="red",
        s=30,
        label="Full Violation Points",
        zorder=5,
    )
    ax.scatter(
        time[partial_violation],
        mean_trace[partial_violation],
        facecolors="none",
        edgecolors="orange",
        s=40,
        label="Partial Violation Points",
        zorder=5,
    )

    # Aesthetics
    ax.set_xlabel("Time")
    ax.set_ylabel("Output")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
