"""Figures for the pdSTL verification suite (see ``src/verification.py``).

Four publication-quality figures, one per verification block. All of them plot
*probability bounds* produced by the exact hard pdSTL semantics -- never a
robustness value, never a smoothed surrogate -- and label them as such.

Styling comes from :mod:`visualization.style`, so these render with the same
fonts, sizing and export settings as every other figure in the project.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from visualization.planning import cov_ellipse_params
from visualization.style import (
    CONFIDENCE_95_K,
    PALETTE,
    figsize,
    save_figure,
    set_ieee_style,
)

__all__ = [
    "plot_always_verification",
    "plot_eventually_verification",
    "plot_stochastic_forward",
    "plot_stochastic_optimization",
]

# Semantic roles, drawn from the shared palette where one already exists.
_MEAN = PALETTE["ego"]["stroke"]        # blue: the belief mean
_BAND = PALETTE["ego"]["fill"]          # blue: its uncertainty band
_LOWER = "#d62728"                      # red: hard lower probability bound
_UPPER = "#1f77b4"                      # blue: hard upper probability bound
_THRESHOLD = "#d62728"                  # red: predicate threshold
_WINDOW = "#ffd92f"                     # yellow: the temporal window
_GOAL = PALETTE["goal"]["stroke"]
_GOAL_FILL = PALETTE["goal"]["fill"]
_SAFE = PALETTE["lane"]["stroke"]
_SAFE_FILL = PALETTE["lane"]["fill"]
_REFERENCE = "#2ca02c"                  # green: external analytic reference
_REJECTED = "#7f7f7f"                   # gray: the independence product (not used)


def _shade_window(ax, interval, label=None):
    """Shade the temporal window ``[a, b]`` behind everything else."""
    a, b = interval
    ax.axvspan(a, b, color=_WINDOW, alpha=0.18, lw=0, zorder=0, label=label)


def _integer_time_axis(ax, times):
    """Discrete time deserves integer ticks, not 2.5 / 7.5."""
    times = np.asarray(times)
    step = 1 if len(times) <= 14 else 2
    ax.set_xticks(times[::step])
    ax.set_xlim(times[0] - 0.5, times[-1] + 0.5)


def _legend_below(ax, *, ncol, fontsize=6.0, y=-0.30):
    """Put a legend under the axes, so it can never sit on top of the data."""
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
        fontsize=fontsize, framealpha=0.95, borderaxespad=0.0,
    )


def _finish(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    written = save_figure(fig, os.path.join(out_dir, stem))
    plt.close(fig)
    return written["pdf"]


# ---------------------------------------------------------------------------
# Verification A -- Eventually
# ---------------------------------------------------------------------------


def plot_eventually_verification(scenario, out_dir):
    """Two stacked panels: the Gaussian state trace, then the union it induces."""
    set_ieee_style("paper")
    fig, (ax_state, ax_prob) = plt.subplots(
        2, 1, sharex=True, figsize=figsize("single", 1.05), constrained_layout=True
    )

    times = scenario["times"]
    mean = scenario["mean_trace"]
    sigma = scenario["sigma_trace"]
    interval = scenario["interval"]
    threshold = scenario["config"]["threshold"]
    probabilities = scenario["atom_probabilities"]
    lower, upper = scenario["bounds"]["recurrent"]
    argmax = scenario["argmax_time"]

    # --- A1: stochastic state trace -------------------------------------
    _shade_window(ax_state, interval)
    ax_state.fill_between(
        times, mean - 2 * sigma, mean + 2 * sigma,
        color=_BAND, alpha=0.25, lw=0, label=r"$\mu_k \pm 2\sigma_k$",
    )
    ax_state.plot(times, mean, color=_MEAN, marker="o", ms=2.5, label=r"$\mu_k$")
    ax_state.axhline(threshold, color=_THRESHOLD, ls="--", lw=1.1,
                     label=rf"$x = {threshold:g}$")
    ax_state.set_ylabel(r"state $x$")
    ax_state.set_title(rf"$\varphi_F = F_{{[{interval[0]},{interval[1]}]}}"
                       rf"(x \geq {threshold:g})$")
    ax_state.legend(loc="upper left", framealpha=0.9)
    ax_state.grid(True)

    # --- A2: atomic probabilities and the resulting union ---------------
    _shade_window(ax_prob, interval, label=rf"window $[{interval[0]},{interval[1]}]$")
    ax_prob.plot(times, probabilities, color=_MEAN, marker="o", ms=2.5,
                 label=r"$p_k = P(X_k \geq 8)$")
    ax_prob.axhline(lower, color=_LOWER, ls="-", lw=1.1,
                    label=rf"$L_F = \max_k p_k = {lower:.3f}$")
    ax_prob.axhline(upper, color=_UPPER, ls="-.", lw=1.1,
                    label=rf"$U_F = \min(1, \sum_k p_k) = {upper:.3f}$")
    ax_prob.fill_between([interval[0] - 0.4, interval[1] + 0.4], lower, upper,
                         color=_UPPER, alpha=0.12, lw=0)
    ax_prob.plot([argmax], [probabilities[argmax]], marker="*", ms=10,
                 color=_LOWER, ls="none", zorder=6,
                 label=rf"$\arg\max_k p_k = {argmax}$")

    # The union structure: each window time is one event that can satisfy phi.
    for k in range(interval[0], interval[1] + 1):
        ax_prob.vlines(k, 0, probabilities[k], color=_MEAN, alpha=0.35, lw=1.0)

    # Bottom-right is the only region the rising curve never enters.
    ax_prob.annotate(
        rf"$[L_F, U_F] = [{lower:.3f},\ {upper:.3f}]$",
        xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": _UPPER, "lw": 0.8, "alpha": 0.95},
    )
    ax_prob.set_xlabel(r"discrete time $k$")
    ax_prob.set_ylabel("probability")
    ax_prob.set_ylim(-0.03, 1.03)
    ax_prob.legend(loc="upper left", fontsize=6.0, framealpha=0.95)
    ax_prob.grid(True)
    _integer_time_axis(ax_prob, times)

    return _finish(fig, out_dir, "eventually")


# ---------------------------------------------------------------------------
# Verification B -- Always
# ---------------------------------------------------------------------------


def plot_always_verification(scenario, out_dir):
    """Three panels: the state band, the conjunction, and the intersection."""
    set_ieee_style("paper")
    # Full text width: the middle panel carries six labelled series, which do
    # not fit legibly in a single column.
    fig, (ax_state, ax_prob, ax_agg) = plt.subplots(
        3, 1, sharex=True, figsize=figsize("double", 0.80), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.85]},
    )

    times = scenario["times"]
    mu = scenario["mu_trace"]
    sigma = scenario["sigma_trace"]
    low, high = scenario["thresholds"]
    a, b = scenario["interval"]
    lower, upper = scenario["bounds"]["recurrent"]

    # --- B1: stochastic state trace with the admissible band ------------
    _shade_window(ax_state, (a, b))
    ax_state.axhspan(low, high, color=_GOAL_FILL, alpha=0.35, lw=0,
                     label=rf"${low:g} < z < {high:g}$")
    ax_state.fill_between(times, mu - 2 * sigma, mu + 2 * sigma,
                          color=_BAND, alpha=0.25, lw=0, label=r"$\mu_k \pm 2\sigma_k$")
    ax_state.plot(times, mu, color=_MEAN, marker="o", ms=2.5, label=r"$\mu_k$")
    ax_state.axhline(low, color=_THRESHOLD, ls="--", lw=0.9)
    ax_state.axhline(high, color=_THRESHOLD, ls="--", lw=0.9)
    ax_state.set_ylabel(r"state $z$")
    ax_state.set_title(rf"$\varphi_G = G_{{[{a},{b}]}}"
                       rf"(({low:g} < z) \wedge (z < {high:g}))$")
    ax_state.legend(loc="upper left", fontsize=6.0, ncol=3, framealpha=0.95)
    # Headroom so the legend sits above the widest uncertainty band.
    ax_state.set_ylim(mu.min() - 2.4 * sigma.max(), mu.max() + 3.4 * sigma.max())
    ax_state.grid(True)

    # --- B2: the two events, the pdSTL band, and the two references -----
    _shade_window(ax_prob, (a, b))
    ax_prob.plot(times, scenario["p_low"], color="#9467bd", marker="^", ms=3,
                 lw=1.0, label=rf"$P(z > {low:g})$")
    ax_prob.plot(times, scenario["p_high"], color="#8c564b", marker="v", ms=3,
                 lw=1.0, label=rf"$P(z < {high:g})$")
    ax_prob.fill_between(times, scenario["band_lower"], scenario["band_upper"],
                         color=_UPPER, alpha=0.20, lw=0,
                         label=r"pdSTL band $[L_k, U_k]$")
    # External analytic reference drawn as a thick underlay, with the Frechet
    # lower bound dashed on top: the two coincide exactly, and drawing them this
    # way is what makes that coincidence visible rather than hidden.
    ax_prob.plot(times, scenario["analytic_band"], color=_REFERENCE, lw=3.2, alpha=0.55,
                 solid_capstyle="round",
                 label=rf"exact $P({low:g}<Z<{high:g})$ (reference)")
    ax_prob.plot(times, scenario["band_lower"], color=_LOWER, lw=1.3, ls="--",
                 label=r"$L_k$ (Frechet lower)")
    ax_prob.plot(times, scenario["band_upper"], color=_UPPER, lw=1.3, ls="-.",
                 label=r"$U_k$ (Frechet upper)")
    ax_prob.plot(times, scenario["product"], color=_REJECTED, ls="--", lw=1.0,
                 marker="x", ms=3.5,
                 label=r"$p_{\rm low}\!\cdot\!p_{\rm high}$ (independence: NOT used)")
    ax_prob.set_ylabel("probability")
    # Headroom below the data for a single-row legend.
    span = scenario["band_lower"].min()
    ax_prob.set_ylim(span - 0.30 * (1.0 - span) - 0.05, 1.01)
    ax_prob.legend(loc="lower left", fontsize=5.8, ncol=3, framealpha=0.95)
    ax_prob.grid(True)

    # --- B3: the temporal intersection --------------------------------
    window = np.arange(a, b + 1)
    window_lower = scenario["band_lower"][a : b + 1]
    window_upper = scenario["band_upper"][a : b + 1]
    m = len(window)

    _shade_window(ax_agg, (a, b))
    ax_agg.vlines(window, window_lower, window_upper, color=_UPPER, lw=6, alpha=0.45)
    ax_agg.plot(window, window_lower, color=_LOWER, marker="o", ms=4, ls="none",
                label=r"$L_k$ per time")
    ax_agg.plot(window, window_upper, color=_UPPER, marker="s", ms=4, ls="none",
                label=r"$U_k$ per time")
    # Clipped to the window: these bounds describe k = a..b, nothing outside it.
    ax_agg.hlines(lower, a - 0.45, b + 0.45, color=_LOWER, lw=1.4,
                  label=rf"$L_G = \max(0, \sum_k L_k - {m - 1}) = {lower:.3f}$")
    ax_agg.hlines(upper, a - 0.45, b + 0.45, color=_UPPER, lw=1.4, ls="-.",
                  label=rf"$U_G = \min_k U_k = {upper:.3f}$")
    ax_agg.fill_between([a - 0.45, b + 0.45], lower, upper, color=_UPPER, alpha=0.14, lw=0)
    ax_agg.set_xlabel(r"discrete time $k$")
    ax_agg.set_ylabel("probability")
    ax_agg.set_ylim(-0.03, 1.28)
    ax_agg.set_title(rf"temporal intersection over $k = {a},\dots,{b}$ "
                     rf"($m = {m}$ distinct events)", fontsize=7.5)
    ax_agg.legend(loc="upper left", fontsize=5.8, ncol=4, framealpha=0.95)
    ax_agg.grid(True)
    _integer_time_axis(ax_agg, times)

    return _finish(fig, out_dir, "always")


# ---------------------------------------------------------------------------
# Verification C -- complete stochastic-system pipeline
# ---------------------------------------------------------------------------


def _draw_region(ax, region, *, facecolor, edgecolor, label, hatch=None, alpha=0.30):
    x_lo, x_hi = region["x"]
    y_lo, y_hi = region["y"]
    ax.add_patch(
        patches.Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
            lw=1.2, hatch=hatch, label=label, zorder=1,
        )
    )


def _draw_geometry(ax, scenario):
    _draw_region(ax, scenario["safe_region"], facecolor=_SAFE_FILL, edgecolor=_SAFE,
                 label="Safe region", alpha=0.35)
    _draw_region(ax, scenario["goal_region"], facecolor=_GOAL_FILL, edgecolor=_GOAL,
                 label="Goal region", alpha=0.55)


def _draw_trajectory(ax, mean, covariance, *, color, label, ellipse_every=2,
                     ellipse_alpha=0.22, ls="-"):
    path = mean[0].detach().numpy()
    covs = covariance[0].detach().numpy()
    ax.plot(path[:, 0], path[:, 1], color=color, ls=ls, marker="o", ms=2.5,
            lw=1.3, label=label, zorder=5)
    for k in range(0, path.shape[0], ellipse_every):
        theta, width, height = cov_ellipse_params(covs[k], k=CONFIDENCE_95_K)
        if width <= 0 or height <= 0:
            continue  # a deterministic initial state has no ellipse to draw
        ax.add_patch(
            patches.Ellipse(
                xy=path[k], width=width, height=height, angle=theta,
                facecolor=color, edgecolor=color, alpha=ellipse_alpha, lw=0.6, zorder=3,
            )
        )
    return path


def plot_stochastic_forward(scenario, out_dir):
    """Three panels: geometry, atomic probability traces, formula-level bounds."""
    set_ieee_style("paper")
    # The workspace is ~10 x 2, so at equal aspect the geometry needs the full
    # text width to itself; the two probability panels share the row below.
    fig = plt.figure(figsize=figsize("double", 0.66), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0])
    ax_geom = fig.add_subplot(grid[0, :])
    ax_prob = fig.add_subplot(grid[1, 0])
    ax_bounds = fig.add_subplot(grid[1, 1])

    horizon = scenario["horizon"]
    times = np.arange(horizon + 1)
    goal_a, goal_b = scenario["goal_interval"]

    # --- C1: geometry ---------------------------------------------------
    _draw_geometry(ax_geom, scenario)
    path = _draw_trajectory(ax_geom, scenario["mean"], scenario["covariance"],
                            color=_MEAN, label=r"mean trajectory $\mu_k$")
    ax_geom.plot(path[0, 0], path[0, 1], marker="s", ms=6, color="black",
                 ls="none", label="start", zorder=6)
    ax_geom.plot(path[-1, 0], path[-1, 1], marker="*", ms=10, color=_LOWER,
                 ls="none", label="end", zorder=6)
    ax_geom.set_xlabel(r"$x$")
    ax_geom.set_ylabel(r"$y$")
    ax_geom.set_title(r"stochastic system, geometry, and $95\%$ covariance ellipses",
                      fontsize=7.5)
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.grid(True)
    _legend_below(ax_geom, ncol=5, fontsize=6.0, y=-0.34)

    # --- C2: atomic (region-level) probability traces --------------------
    safe = scenario["safe_trace"].detach().numpy()
    goal = scenario["goal_trace"].detach().numpy()
    ax_prob.axvspan(0, horizon, color=_WINDOW, alpha=0.18, lw=0,
                    label=rf"$G$ window $[0,{horizon}]$")
    ax_prob.axvspan(goal_a, goal_b, color=_GOAL_FILL, alpha=0.30, lw=0,
                    label=rf"$F$ window $[{goal_a},{goal_b}]$")
    ax_prob.fill_between(times, safe[:, 0], safe[:, 1], color=_SAFE, alpha=0.30, lw=0)
    ax_prob.plot(times, safe[:, 0], color=_SAFE, lw=1.3, marker="o", ms=2.5,
                 label="Safe $[L_k, U_k]$")
    ax_prob.fill_between(times, goal[:, 0], goal[:, 1], color=_GOAL, alpha=0.30, lw=0)
    ax_prob.plot(times, goal[:, 0], color=_GOAL, lw=1.3, marker="^", ms=3,
                 label="Goal $[L_k, U_k]$")
    ax_prob.set_xlabel(r"discrete time $k$")
    ax_prob.set_ylabel("probability")
    ax_prob.set_ylim(-0.03, 1.03)
    ax_prob.set_title("atomic region probabilities", fontsize=7.5)
    ax_prob.legend(loc="center left", fontsize=5.8, framealpha=0.95)
    ax_prob.grid(True)
    _integer_time_axis(ax_prob, times)

    # --- C3: formula-level bounds ---------------------------------------
    labels = [r"$G\,\mathrm{Safe}$", r"$F\,\mathrm{Goal}$",
              r"$G\,\mathrm{Safe} \wedge F\,\mathrm{Goal}$"]
    keys = ["G Safe", "F Goal", "G Safe AND F Goal"]
    positions = np.arange(len(keys))

    for pos, key in zip(positions, keys):
        lower, upper = scenario["results"][key]["recurrent"]
        ax_bounds.barh(pos, upper - lower, left=lower, height=0.45,
                       color=_UPPER, alpha=0.35, edgecolor=_UPPER, lw=1.0)
        ax_bounds.plot([lower], [pos], marker="|", ms=14, color=_LOWER, mew=2.0)
        ax_bounds.plot([upper], [pos], marker="|", ms=14, color=_UPPER, mew=2.0)
        ax_bounds.text(upper + 0.03, pos, f"[{lower:.3f}, {upper:.3f}]",
                       va="center", fontsize=6.0)

    ax_bounds.set_yticks(positions)
    ax_bounds.set_yticklabels(labels, fontsize=7)
    ax_bounds.set_xlim(0.0, 1.42)
    ax_bounds.set_xticks([0.0, 0.5, 1.0])
    ax_bounds.set_xlabel("hard probability bound")
    ax_bounds.set_title("formula-level enclosures", fontsize=7.5)
    ax_bounds.invert_yaxis()
    ax_bounds.grid(True, axis="x")
    # ASCII only: the serif/STIX mathtext set has no glyph for a check mark.
    ax_bounds.annotate(
        "backends agree: reference $=$ compiled $=$ recurrent",
        xy=(0.5, -0.30), xycoords="axes fraction", ha="center", va="top",
        fontsize=6.0, color=_REFERENCE,
    )

    return _finish(fig, out_dir, "stochastic_forward")


def plot_stochastic_optimization(scenario, result, out_dir):
    """Three panels: trajectories before/after, bound history, gradient norm."""
    set_ieee_style("paper")
    fig = plt.figure(figsize=figsize("double", 0.66), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0], width_ratios=[1.15, 0.85])
    ax_geom = fig.add_subplot(grid[0, :])
    ax_bound = fig.add_subplot(grid[1, 0])
    ax_grad = fig.add_subplot(grid[1, 1])

    history = result["history"]
    iterations = np.arange(result["iterations"])

    # --- 1: initial vs optimized trajectory ------------------------------
    _draw_geometry(ax_geom, scenario)
    _draw_trajectory(ax_geom, result["initial_mean"], result["initial_covariance"],
                     color=_REJECTED, label="initial", ls="--", ellipse_alpha=0.14)
    path = _draw_trajectory(ax_geom, result["final_mean"], result["final_covariance"],
                            color=_MEAN, label="optimized")
    ax_geom.plot(path[0, 0], path[0, 1], marker="s", ms=6, color="black", ls="none", zorder=6)
    ax_geom.set_xlabel(r"$x$")
    ax_geom.set_ylabel(r"$y$")
    ax_geom.set_title("mean trajectory before / after maximizing "
                      r"$P_{\mathrm{lower}}(\varphi)$", fontsize=7.5)
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.grid(True)
    _legend_below(ax_geom, ncol=4, fontsize=6.0, y=-0.34)

    # --- 2: the optimized quantity itself --------------------------------
    ax_bound.plot(iterations, history["lower"], color=_LOWER, lw=1.5,
                  label=r"hard $P_{\mathrm{lower}}(\varphi)$  (optimized)")
    ax_bound.plot(iterations, history["upper"], color=_UPPER, lw=1.2, ls="-.",
                  label=r"hard $P_{\mathrm{upper}}(\varphi)$")
    ax_bound.fill_between(iterations, history["lower"], history["upper"],
                          color=_UPPER, alpha=0.12, lw=0)
    ax_bound.annotate(
        rf"$P_{{\mathrm{{lower}}}}: {history['lower'][0]:.3f} \rightarrow "
        rf"{history['lower'][-1]:.3f}$",
        xy=(0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": _LOWER, "lw": 0.8, "alpha": 0.95},
    )
    ax_bound.set_xlabel("Adam iteration")
    ax_bound.set_ylabel("hard probability bound")
    ax_bound.set_ylim(-0.03, 1.03)
    ax_bound.set_title(r"direct maximization of the hard lower bound", fontsize=7.5)
    ax_bound.legend(loc="center right", fontsize=5.8, framealpha=0.9)
    ax_bound.grid(True)

    # --- 3: gradient norm ------------------------------------------------
    ax_grad.semilogy(iterations, np.maximum(history["grad_norm"], 1e-16),
                     color=_MEAN, lw=1.3)
    ax_grad.set_xlabel("Adam iteration")
    ax_grad.set_ylabel(r"$\|\nabla_v P_{\mathrm{lower}}\|$")
    ax_grad.set_title("gradient norm", fontsize=7.5)
    ax_grad.grid(True, which="both")

    return _finish(fig, out_dir, "stochastic_optimization")
