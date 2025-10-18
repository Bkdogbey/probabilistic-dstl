import matplotlib.pyplot as plt
import numpy as np


def plot_bounds_with_trace(
    time, mean_trace, var_trace, lower_bound=None, upper_bound=None, threshold=50
):
    std = np.sqrt(var_trace)

    violate_mean = mean_trace < threshold
    violate_1sigma = (mean_trace - std) < threshold

    plt.figure(figsize=(12, 6))
    plt.plot(time, mean_trace, label="Mean Height", color="blue")
    plt.fill_between(
        time,
        mean_trace - std,
        mean_trace + std,
        color="blue",
        alpha=0.2,
        label="1-sigma Interval",
    )
    plt.axhline(
        threshold, color="red", linestyle="--", label=f"Threshold Height = {threshold}m"
    )

    plt.fill_between(time, 0, threshold, where=violate_mean, color="red", alpha=0.1)
    plt.fill_between(
        time, 0, threshold, where=violate_1sigma, color="orange", alpha=0.1
    )
    plt.scatter(
        time[violate_mean],
        mean_trace[violate_mean],
        color="red",
        s=30,
        label="Mean Violation",
    )
    plt.scatter(
        time[violate_1sigma],
        mean_trace[violate_1sigma],
        facecolors="none",
        edgecolors="orange",
        s=40,
        label="Partial Violation",
    )

    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.show()
