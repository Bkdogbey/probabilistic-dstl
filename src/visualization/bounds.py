import matplotlib.pyplot as plt
import numpy as np


def plot_bounds_with_trace(
    time, mean_trace, var_trace, lower_bound, upper_bound, threshold=50
):
    std = np.sqrt(var_trace)

    lower_bound = mean_trace - std
    upper_bound = mean_trace + std

    violate_full = (lower_bound < threshold) & (upper_bound < threshold)

    violate_partial = (lower_bound < threshold) & ~violate_full

    plt.figure(figsize=(12, 6))

    plt.plot(time, mean_trace, label="Mean Height", color="blue", linewidth=2)
    plt.fill_between(
        time,
        mean_trace - std,
        mean_trace + std,
        color="blue",
        alpha=0.2,
        label="±1 Sigma Interval",
    )

    plt.axhline(
        threshold, color="red", linestyle="--", label=f"Threshold Height = {threshold}m"
    )

    y_max = np.max(mean_trace + std) * 1.1
    y_min = np.min(mean_trace - std) * 0.9

    plt.fill_between(
        time,
        y_min,
        y_max,
        where=violate_full,
        color="red",
        alpha=0.1,
        label="Full violation (both bounds < threshold)",
    )

    plt.fill_between(
        time,
        y_min,
        y_max,
        where=violate_partial,
        color="orange",
        alpha=0.1,
        label="Partial violation (lower bound < threshold)",
    )

    plt.scatter(
        time[violate_full],
        mean_trace[violate_full],
        color="red",
        s=30,
        label="Full Violation Points",
        zorder=5,
    )
    plt.scatter(
        time[violate_partial],
        mean_trace[violate_partial],
        facecolors="none",
        edgecolors="orange",
        s=40,
        label="Partial Violation Points",
        zorder=5,
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Height [m]")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
