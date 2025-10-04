import matplotlib.pyplot as plt
import numpy as np


def plot_bounds_with_trace(time, mean_trace, var_trace, lower_bound, upper_bound):
    plt.figure(figsize=(12, 6))
    plt.plot(time, mean_trace, label="Mean Height", color="blue")
    plt.fill_between(
        time,
        mean_trace - np.sqrt(var_trace),
        mean_trace + np.sqrt(var_trace),
        color="blue",
        alpha=0.2,
        label="1-sigma Interval",
    )
    plt.axhline(50, color="red", linestyle="--", label="Threshold Height = 50m")
    plt.title("Height Trajectory with 1-sigma Confidence Interval")
    plt.xlabel("Time [s]")
    plt.ylabel("Height [m]")
    plt.legend()
    plt.grid()
    plt.show()
