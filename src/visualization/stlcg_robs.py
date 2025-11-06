import matplotlib.pyplot as plt
import numpy as np


def plot_rnn_robustness(time, rob_lower_rnn, rob_upper_rnn, formula, threshold):
    """
    Plot RNN temporal STL robustness bounds.

    Parameters
    ----------
    time : array_like
        Array of time values.
    rob_lower_rnn : array_like
        Lower bound of RNN temporal robustness (from Always operator).
    rob_upper_rnn : array_like
        Upper bound of RNN temporal robustness (from Always operator).
    formula : STL_Formula
        The STL formula object.
    threshold : float
        Threshold value for the STL predicate .
    """

    formula_str = str(formula)

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # RNN TEMPORAL ROBUSTNESS =
    ax.plot(
        time,
        rob_lower_rnn,
        label="Lower Robustness (pessimistic)",
        color="darkred",
        linewidth=2.5,
        linestyle="-",
    )
    ax.plot(
        time,
        rob_upper_rnn,
        label="Upper Robustness (optimistic)",
        color="darkgreen",
        linewidth=2.5,
        linestyle="-",
    )

    # Fill uncertainty band
    ax.fill_between(
        time,
        rob_lower_rnn,
        rob_upper_rnn,
        color="purple",
        alpha=0.3,
        label="Robustness Uncertainty",
    )

    # Zero line (satisfaction boundary)
    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=2,
        label="Satisfaction Boundary (ρ=0)",
    )

    # Fill satisfaction/violation regions
    ax.fill_between(
        time,
        0,
        rob_lower_rnn,
        where=(rob_lower_rnn >= 0),
        color="green",
        alpha=0.2,
        label="Definitely Satisfied",
        interpolate=True,
    )
    ax.fill_between(
        time,
        rob_upper_rnn,
        0,
        where=(rob_upper_rnn < 0),
        color="red",
        alpha=0.2,
        label="Definitely Violated",
        interpolate=True,
    )
    ax.fill_between(
        time,
        rob_lower_rnn,
        rob_upper_rnn,
        where=((rob_lower_rnn < 0) & (rob_upper_rnn >= 0)),
        color="yellow",
        alpha=0.3,
        label="Uncertain Region",
        interpolate=True,
    )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("STL Robustness ρ(t)", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    title = "RNN Temporal STL Robustness"
    if formula_str:
        title += f"\n{formula_str}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()
