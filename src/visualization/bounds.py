import matplotlib.pyplot as plt
import numpy as np


def plot_bounds_with_trace(time, mean_trace, var_trace, threshold=50):
    std = np.sqrt(var_trace)
    lower_1sigma = mean_trace - std

    violate_mean = mean_trace < threshold
    violate_1sigma = lower_1sigma < threshold

    plt.figure(figsize=(12, 6))
    plt.plot(time, mean_trace, label="Mean Height", color="blue", linewidth=2)
    plt.fill_between(
        time,
        mean_trace - std,
        mean_trace + std,
        color="blue",
        alpha=0.2,
        label="1-Sigma Interval",
    )

    plt.axhline(
        threshold, color="red", linestyle="--", label=f"Threshold Height = {threshold}m"
    )

    # Get y-axis limits for vertical bands
    y_max = np.max(mean_trace + std) * 1.1
    y_min = np.min(mean_trace - std) * 0.9

    # Vertical bands for mean violations (red)
    plt.fill_between(
        time,
        y_min,
        y_max,
        where=violate_mean,
        color="red",
        alpha=0.1,
        label="Mean < threshold (violation)",
    )

    only_1sigma = violate_1sigma & ~violate_mean
    plt.fill_between(
        time,
        y_min,
        y_max,
        where=only_1sigma,
        color="orange",
        alpha=0.1,
        label="Lower Bound < Stl threshold (risk)",
    )

    plt.scatter(
        time[violate_mean],
        mean_trace[violate_mean],
        color="red",
        s=30,
        label="STL Violation Points",
        zorder=5,
    )
    plt.scatter(
        time[only_1sigma],
        mean_trace[only_1sigma],
        facecolors="none",
        edgecolors="orange",
        s=40,
        label="Partial STL Violation Points",
        zorder=5,
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Height [m]")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
