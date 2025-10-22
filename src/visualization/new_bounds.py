import matplotlib.pyplot as plt
import numpy as np


def plot_bounds(
    time, mean_trace, var_trace, threshold=50
):
    std = np.sqrt(var_trace)
    lower_1sigma = mean_trace - std

    violate_mean = mean_trace < threshold
    violate_1sigma = lower_1sigma < threshold

    plt.figure(figsize=(12, 6))
    plt.plot(
        time, mean_trace, label="Mean Height", color="black", linewidth=2, zorder=10
    )

    plt.fill_between(
        time,
        mean_trace - std,
        mean_trace + std,
        color="blue",
        alpha=0.2,
        label="Safe Region",
        linewidth=0,
    )
    only_1sigma = violate_1sigma & ~violate_mean
    plt.fill_between(
        time,
        mean_trace - std,
        mean_trace + std,
        where=only_1sigma,
        color="orange",
        alpha=0.3,
        label="Partial violation",
        linewidth=0,
        interpolate=True,
    )
    plt.fill_between(
        time,
        mean_trace - std,
        mean_trace + std,
        where=violate_mean,
        color="red",
        alpha=0.3,
        label="Full violation",
        linewidth=0,
        interpolate=True,
    )
    plt.axhline(
        threshold,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold}m",
        zorder=5,
    )
    plt.scatter(
        time[violate_mean],
        mean_trace[violate_mean],
        color="red",
        s=30,
        label="Mean Violation Points",
        zorder=5,
    )
    plt.scatter(
        time[only_1sigma],
        mean_trace[only_1sigma],
        facecolors="none",
        edgecolors="orange",
        s=40,
        label="Risk Points",
        zorder=5,
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Height [m]")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
