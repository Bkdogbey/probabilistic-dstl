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

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # ========== TRACE PLOT ==========
    # Main trace and uncertainty
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
        label=f"Threshold = {threshold} m",
    )
    # Vertical range for shading violations
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
        label="Full violation (both bounds < threshold)",
    )
    ax1.fill_between(
        time,
        y_min,
        y_max,
        where=partial_violation,
        color="orange",
        alpha=0.1,
        label="Partial violation (lower bound < threshold)",
    )
    # Violation markers
    ax1.scatter(
        time[full_violation],
        mean_trace[full_violation],
        color="red",
        s=30,
        label="Full Violation Points",
        zorder=5,
    )
    ax1.scatter(
        time[partial_violation],
        mean_trace[partial_violation],
        facecolors="none",
        edgecolors="orange",
        s=40,
        label="Partial Violation Points",
        zorder=5,
    )
    # Aesthetics
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Output")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Trajectory with Violation Regions")

    # ROBUSTNESS PLOT 
    # Compute pointwise robustness: ρ(t) = Lower sigma bound - threshold
    robustness = lower_sigma - threshold

    ax2.plot(
        time, robustness, label="Robustness ρ(t) = x(t) - h", color="blue", linewidth=2
    )

    # Zero line (satisfaction boundary)
    ax2.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=2,
        label="Satisfaction Boundary (ρ=0)",
    )

    # Fill regions
    ax2.fill_between(
        time,
        0,
        robustness,
        where=(robustness >= 0),
        color="green",
        alpha=0.2,
        label="Satisfied (ρ>0)",
        interpolate=True,
    )
    ax2.fill_between(
        time,
        robustness,
        0,
        where=(robustness < 0),
        color="red",
        alpha=0.2,
        label="Violated (ρ<0)",
        interpolate=True,
    )

    # Compute STL robustness (minimum over all time)
    rho_stl = float(np.min(robustness))
    t_critical = float(time[np.argmin(robustness)])

    # Mark critical point
    ax2.axvline(t_critical, color="darkred", linestyle=":", linewidth=2)
    ax2.scatter(
        [t_critical],
        [rho_stl],
        color="darkred",
        s=150,
        zorder=10,
        marker="o",
        edgecolors="black",
        linewidths=2,
    )

    # Status text
    status = "SATISFIED" if rho_stl >= 0 else "VIOLATED"
    status_color = "green" if rho_stl >= 0 else "red"
    ax2.text(
        0.02,
        0.98,
        f"STL Robustness: ρ = {rho_stl:.3f}\nStatus: {status}",
        transform=ax2.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="top",
        color=status_color,
        bbox=dict(
            boxstyle="round", facecolor="white", edgecolor=status_color, linewidth=2
        ),
    )

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Robustness ρ(t)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("STL Robustness")

    plt.tight_layout()
    plt.show()
