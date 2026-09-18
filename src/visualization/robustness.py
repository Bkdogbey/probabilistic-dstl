"""Plots for the scalar pdSTL cases; a formula trace of K origins is drawn at the first K timestamps."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Tableau 10
_BLUE = "#1f77b4"
_RED = "#d62728"
_GREEN = "#2ca02c"
_GRAY = "#7f7f7f"

# Endpoints closer than this are coincident for display purposes; Gaussian
# atoms return [p, p], whose fill between endpoints would be invisible.
_COINCIDENT = 1e-9


def _to_numpy(trace):
    """Return a probability-bound trace as numpy [K, 2], whatever its length."""
    if isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()
    trace = np.asarray(trace)

    if trace.ndim == 3:  # [B, K, 2] -> first batch element
        trace = trace[0]
    elif trace.ndim == 1:  # exact scalar series -> coincident endpoints
        trace = np.stack([trace, trace], axis=-1)

    if trace.ndim != 2 or trace.shape[-1] != 2:
        raise ValueError(f"Expected a [K, 2] bound trace, got {trace.shape}")
    return trace


def _finish(fig, save_path=None, show=False):
    """Save and/or display, so runs can be fully non-interactive."""
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def _draw_bounds(ax, time, bounds, label_lower, label_upper):
    """Draw a [K, 2] bound trace; coincident endpoints become a single marked line."""
    k = len(bounds)
    t = np.asarray(time)[:k]
    lower, upper = bounds[:, 0], bounds[:, 1]
    coincident = bool(np.all(np.abs(upper - lower) <= _COINCIDENT))

    # A single origin has no line to draw, so always mark the points.
    marker = "o"

    if coincident:
        ax.plot(t, lower, color=_BLUE, lw=1.8, marker=marker, ms=5,
                label=f"{label_lower} = {label_upper}")
    else:
        ax.fill_between(t, lower, upper, alpha=0.25, color=_GREEN)
        ax.plot(t, lower, color=_BLUE, lw=1.8, marker=marker, ms=4, label=label_lower)
        ax.plot(t, upper, color=_RED, lw=1.8, marker=marker, ms=4, label=label_upper)

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return k


def _mark_absent(ax, time, k):
    """Shade origins that have no complete window, so they read as absent."""
    time = np.asarray(time)
    if k >= len(time):
        return
    # Start midway to the next step so the last valid origin's marker stays clear.
    start = (time[k - 1] + time[k]) / 2 if k else time[0]
    ax.axvspan(start, time[-1], color=_GRAY, alpha=0.10)
    ax.text(
        (start + time[-1]) / 2,
        0.5,
        "no complete\nwindow",
        ha="center", va="center", fontsize=8, color=_GRAY,
    )


def plot_case(
    time,
    formula_trace,
    mean_trace=None,
    var_trace=None,
    lower_trace=None,
    upper_trace=None,
    predicate_traces=None,
    thresholds=None,
    formula_str=None,
    title=None,
    figsize=(9, 8),
    save_path=None,
    show=False,
):
    """Three panels for one case: state, atom probabilities, formula interval."""
    time = np.asarray(time)
    formula = _to_numpy(formula_trace)

    n_panels = 3 if predicate_traces else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    ax_state = axes[0]
    ax_pred = axes[1] if predicate_traces else None
    ax_form = axes[-1]

    # (a) state and its uncertainty
    if lower_trace is not None and upper_trace is not None:
        lower, upper = np.asarray(lower_trace), np.asarray(upper_trace)
        if var_trace is not None:
            sigma = np.sqrt(np.maximum(np.asarray(var_trace), 0.0))
            ax_state.fill_between(
                time, lower - sigma, upper + sigma, alpha=0.12, color=_GRAY,
                zorder=1, label=r"$\pm$ residual std",
            )
        ax_state.fill_between(
            time, lower, upper, alpha=0.25, color=_BLUE, zorder=2,
            label="descriptor bounds [lower, upper]",
        )
        ax_state.plot(
            time, (lower + upper) / 2, color=_BLUE, lw=1.8, marker="o", ms=3,
            zorder=3, label="descriptor midpoint",
        )
    elif mean_trace is not None and var_trace is not None:
        mean = np.asarray(mean_trace)
        sigma = np.sqrt(np.maximum(np.asarray(var_trace), 0.0))
        ax_state.fill_between(
            time, mean - sigma, mean + sigma, alpha=0.25, color=_BLUE,
            label=r"state uncertainty band $\mu \pm \sigma$",
        )
        ax_state.plot(time, mean, color=_BLUE, lw=1.8, marker="o", ms=3, label=r"$\mu(t)$")

    if (mean_trace is not None or lower_trace is not None) and thresholds is not None:
        for th in np.atleast_1d(thresholds):
            ax_state.axhline(float(th), color=_RED, ls="--", lw=1.3)
        ax_state.plot([], [], color=_RED, ls="--", lw=1.3, label="threshold")

    ax_state.set_ylabel("state $x$")
    ax_state.set_title("(a) Predicted state", loc="left", fontweight="bold")
    ax_state.legend(fontsize=8, loc="best", framealpha=0.95)
    ax_state.grid(True, alpha=0.3)

    # (b) atomic event probabilities
    if ax_pred is not None:
        for label, tr in predicate_traces.items():
            bounds = _to_numpy(tr)
            t = time[: len(bounds)]
            ax_pred.plot(t, bounds[:, 0], lw=1.6, marker="o", ms=3, label=label)
        ax_pred.set_ylabel("probability")
        ax_pred.set_ylim(-0.05, 1.05)
        ax_pred.set_title(
            "(b) Atomic event probability (exact, endpoints coincide)",
            loc="left", fontweight="bold",
        )
        ax_pred.legend(fontsize=8, loc="best", framealpha=0.95)
        ax_pred.grid(True, alpha=0.3)

    # (c) formula interval at its valid origins only
    k = _draw_bounds(ax_form, time, formula, "lower", "upper")
    _mark_absent(ax_form, time, k)
    ax_form.set_ylabel("stochastic robustness")
    ax_form.set_xlabel("time [s]")
    ax_form.set_title(
        f"(c) Formula interval at {k} valid origin{'s' if k != 1 else ''}",
        loc="left", fontweight="bold",
    )
    ax_form.legend(fontsize=8, loc="best", framealpha=0.95)

    heading = title or formula_str
    if heading:
        fig.suptitle(heading, fontsize=11, fontweight="bold")

    _finish(fig, save_path, show)
    return fig, axes


def plot_synthesis(
    time,
    initial,
    optimized,
    objective_history,
    thresholds=None,
    title=None,
    figsize=(10, 7),
    save_path=None,
    show=False,
):
    """Initial vs optimized state and formula traces, plus the objective history."""
    time = np.asarray(time)
    fig, ((ax_state, ax_form), (ax_obj, ax_blank)) = plt.subplots(
        2, 2, figsize=figsize
    )
    ax_blank.axis("off")

    for run, color, name in ((initial, _GRAY, "initial"), (optimized, _BLUE, "optimized")):
        if "lower" in run and "upper" in run:
            lower, upper = np.asarray(run["lower"]), np.asarray(run["upper"])
            if run.get("var") is not None:
                sigma = np.sqrt(np.maximum(np.asarray(run["var"]), 0.0))
                ax_state.fill_between(
                    time, lower - sigma, upper + sigma, alpha=0.08, color=color
                )
            ax_state.fill_between(time, lower, upper, alpha=0.18, color=color)
            ax_state.plot(
                time, (lower + upper) / 2, color=color, lw=1.8, marker="o", ms=3,
                label=name,
            )
        else:
            mean = np.asarray(run["mean"])
            sigma = np.sqrt(np.maximum(np.asarray(run["var"]), 0.0))
            ax_state.fill_between(time, mean - sigma, mean + sigma, alpha=0.18, color=color)
            ax_state.plot(time, mean, color=color, lw=1.8, marker="o", ms=3, label=name)

        bounds = _to_numpy(run["formula"])
        t = time[: len(bounds)]
        ax_form.plot(t, bounds[:, 0], color=color, lw=1.8, marker="o", ms=5, label=f"{name} lower")
        if not np.allclose(bounds[:, 0], bounds[:, 1], atol=_COINCIDENT):
            ax_form.plot(t, bounds[:, 1], color=color, lw=1.2, ls="--", marker="o",
                         ms=4, label=f"{name} upper")

    for th in np.atleast_1d(thresholds) if thresholds is not None else []:
        ax_state.axhline(float(th), color=_RED, ls="--", lw=1.3)

    ax_state.set_xlabel("time [s]")
    ax_state.set_ylabel("state $x$")
    ax_state.set_title("(a) State (band = descriptor bounds or $\\mu \\pm \\sigma$)",
                       loc="left", fontweight="bold")
    ax_state.legend(fontsize=8)
    ax_state.grid(True, alpha=0.3)

    ax_form.set_xlabel("time [s]")
    ax_form.set_ylabel("stochastic robustness")
    ax_form.set_ylim(-0.05, 1.05)
    # Keep the full horizon on the axis; a single valid origin would otherwise
    # autoscale to a meaningless window around one point.
    ax_form.set_xlim(time[0] - 0.05, time[-1] + 0.05)
    ax_form.set_title("(b) Formula interval (direct)", loc="left", fontweight="bold")
    ax_form.legend(fontsize=8)
    ax_form.grid(True, alpha=0.3)

    ax_obj.plot(objective_history, color=_GREEN, lw=1.5)
    ax_obj.set_xlabel("iteration")
    ax_obj.set_ylabel("objective $J$")
    ax_obj.set_title("(c) Optimisation objective", loc="left", fontweight="bold")
    ax_obj.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    _finish(fig, save_path, show)
    return fig, (ax_state, ax_form, ax_obj)


def plot_end_to_end(
    time,
    initial,
    optimized,
    threshold,
    title=None,
    figsize=(9, 6),
    save_path=None,
    show=False,
):
    """Initial vs optimized mean +/- sigma and atomic probability for the end-to-end reach."""
    time = np.asarray(time)
    fig, (ax_state, ax_prob) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for run, color, name in ((initial, _GRAY, "initial"), (optimized, _BLUE, "optimized")):
        mean = np.asarray(run["mean"])
        sigma = np.sqrt(np.maximum(np.asarray(run["var"]), 0.0))
        ax_state.fill_between(time, mean - sigma, mean + sigma, alpha=0.18, color=color)
        ax_state.plot(
            time, mean, color=color, lw=1.8, marker="o", ms=3,
            label=rf"{name} $\mu_x$ ($\pm\sigma$ band)",
        )

        atomic = _to_numpy(run["atomic"])
        ax_prob.plot(
            time[: len(atomic)], atomic[:, 0], color=color, lw=1.8, marker="o", ms=3,
            label=f"{name}: R = {run['score']:.4f}",
        )

    ax_state.axhline(float(threshold), color=_RED, ls="--", lw=1.3, label=f"x = {threshold}")
    ax_state.set_ylabel("state $x$")
    ax_state.set_title("(a) Predicted mean and uncertainty", loc="left", fontweight="bold")
    ax_state.legend(fontsize=8, loc="best", framealpha=0.95)
    ax_state.grid(True, alpha=0.3)

    ax_prob.set_ylabel("probability")
    ax_prob.set_xlabel("time [s]")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_title(
        rf"(b) Atomic probability $p_k = P(X_k \geq {threshold})$", loc="left",
        fontweight="bold",
    )
    ax_prob.legend(fontsize=8, loc="best", framealpha=0.95)
    ax_prob.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    _finish(fig, save_path, show)
    return fig, (ax_state, ax_prob)


def plot_mpc_reach(
    dt,
    result,
    threshold,
    dim=0,
    title=None,
    figsize=(9, 8),
    save_path=None,
    show=False,
):
    """Receding-horizon run: executed state, per-window plans, scores and applied controls."""
    executed = np.asarray([state[0][dim].item() for state in result.states])
    controls = _to_array(result.applied_controls)
    scores = np.asarray([plan.hard_interval[0] for plan in result.window_plans])
    time = dt * np.arange(len(executed))

    fig, (ax_state, ax_score, ax_u) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for k, plan in enumerate(result.window_plans):
        plan_x = _to_array(plan.rollout.aux["mean_trace"])[0, :, dim]
        ax_state.plot(
            dt * (k + np.arange(len(plan_x))), plan_x, color=_BLUE, lw=0.8, alpha=0.25,
            label="predicted H-step plans" if k == 0 else None,
        )
    ax_state.plot(time, executed, color="black", lw=2.0, marker="o", ms=3, label="executed x")
    ax_state.axhline(float(threshold), color=_RED, ls="--", lw=1.3, label=f"x = {threshold}")
    ax_state.set_ylabel("state $x$")
    ax_state.set_title(
        f"(a) Executed trajectory ({result.stopped_reason})", loc="left", fontweight="bold"
    )
    ax_state.legend(fontsize=8, loc="best", framealpha=0.95)
    ax_state.grid(True, alpha=0.3)

    ax_score.plot(time[: len(scores)], scores, color=_GREEN, lw=1.8, marker="o", ms=3)
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.set_ylabel("pdSTL score")
    ax_score.set_title("(b) Lower robustness of each replanned window", loc="left", fontweight="bold")
    ax_score.grid(True, alpha=0.3)

    t_u = time[: len(controls)]
    ax_u.step(t_u, controls[:, 0], where="post", color=_BLUE, lw=1.6, label="$v_x$ applied")
    ax_u.step(t_u, controls[:, 1], where="post", color=_GRAY, lw=1.6, label="$v_y$ applied")
    ax_u.set_ylabel("control")
    ax_u.set_xlabel("time [s]")
    ax_u.set_title("(c) First control of each plan", loc="left", fontweight="bold")
    ax_u.legend(fontsize=8, loc="best", framealpha=0.95)
    ax_u.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    _finish(fig, save_path, show)
    return fig, (ax_state, ax_score, ax_u)


def _to_array(value):
    """Detach a tensor (or pass an array through) as numpy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def plot_end_to_end_plans(result, initial, atom, cfg, *, show=False, save_path=None):
    def presentation(plan):
        return {
            "mean": plan.rollout.aux["mean_trace"][0, :, cfg["dim"]],
            "var": plan.rollout.aux["cov_trace"][0, :, cfg["dim"], cfg["dim"]],
            "atomic": atom(plan.rollout.belief_trajectory),
            "score": plan.hard_interval[0],
        }
    return plot_end_to_end(
        [k * cfg["dt"] for k in range(cfg["H"] + 1)],
        presentation(initial), presentation(result), cfg["threshold"],
        show=show, save_path=save_path,
    )
