import os
from itertools import cycle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch
from types import SimpleNamespace

PALETTE = {  # Tableau 10
    "ego": {"fill": "#1f77b4", "stroke": "#1f77b4"},
    "plan": {"fill": "#ff7f0e", "stroke": "#ff7f0e"},
    "visit": {"fill": "#c5b0d5", "stroke": "#9467bd"},
    "obs_static": {"fill": "#ff9896", "stroke": "#d62728"},
    "obs_moving": {"fill": "#ff9896", "stroke": "#d62728"},
    "lane": {"fill": "#c7c7c7", "stroke": "#7f7f7f"},
    "goal": {"fill": "#98df8a", "stroke": "#2ca02c"},
    "road": {"fill": "#F2F2F7"},
}


def geometry(environment):
    """Adapt an Environment's named regions to the flat shapes the drawing code uses.

    The translation lives here rather than on Environment: plotting is a consumer of the
    geometry, so it is plotting's job to read regions in the form it wants.
    """
    from planning.environment import RectangleRegion
    from planning.environment import CircleRegion, MovingRectangleRegion

    if not hasattr(environment, "regions"):
        return environment  # already a flat view

    def rectangles(role):
        return [
            {"x": r.x, "y": r.y, "name": r.name, "style": r.style}
            for r in environment.by_role(role)
            if isinstance(r, RectangleRegion)
        ]

    metadata = getattr(environment, "metadata", {})
    workspace = next(iter(environment.by_role("workspace")), None)
    goal = next(iter(environment.by_role("goal")), None)
    return SimpleNamespace(
        bounds={"x": workspace.x, "y": workspace.y, "style": workspace.style} if workspace is not None else None,
        goal={"x": goal.x, "y": goal.y, "style": goal.style} if goal is not None else None,
        obstacles=rectangles("obstacle"),
        visit_regions=rectangles("visit"),
        circle_obstacles=[
            {"center": r.center, "radius": r.radius}
            for r in environment.by_role("obstacle")
            if isinstance(r, CircleRegion)
        ],
        moving_obstacles=[
            {
                "x_traj": r.centers[..., 0], "y_traj": r.centers[..., 1],
                "width": r.width, "height": r.height,
            }
            for r in environment.by_role("obstacle")
            if isinstance(r, MovingRectangleRegion)
        ],
        lane_markings=metadata.get("lane_markings", []),
        success=metadata.get("success"),
        label=metadata.get("label", ""),
        plot_xlim=metadata.get("plot_xlim"),
        robot_dims=metadata.get("robot_dims"),
    )


def cov_ellipse_params(cov, k=1.96):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * k * np.sqrt(vals)
    return theta, width, height


def plot_covariance_ellipse(
    ax,
    mean,
    cov,
    k=1.96,
    facecolor="blue",
    edgecolor="blue",
    alpha=0.4,
    zorder=10,
    label=None,
):
    """Draws a confidence ellipse for a 2D Gaussian belief."""
    theta, width, height = cov_ellipse_params(cov, k)

    ellipse = patches.Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    ax.add_patch(ellipse)


def draw_road_backdrop(ax, env):
    env = geometry(env)
    """Road, goal lane and lane markings; returns (road_lo, road_hi)."""
    road_lo = min(lm["y"] for lm in env.lane_markings) if env.lane_markings else -2.0
    road_hi = max(lm["y"] for lm in env.lane_markings) if env.lane_markings else 6.0
    ax.axhspan(road_lo, road_hi, color=PALETTE["road"]["fill"], zorder=0)
    if env.goal:
        gy0, gy1 = env.goal["y"]
        ax.axhspan(
            gy0,
            gy1,
            color=PALETTE["goal"]["fill"],
            alpha=0.22,
            zorder=1,
            label="Goal Lane",
        )
    for lane in env.lane_markings:
        style = "--" if lane["style"] == "dashed" else "-"
        lw = 1.5 if lane["style"] == "dashed" else 2.0
        ax.axhline(
            lane["y"],
            color=PALETTE["lane"]["stroke"],
            linestyle=style,
            linewidth=lw,
            alpha=0.9,
            zorder=2,
        )
    return road_lo, road_hi


def _region_style(region, **defaults):
    style = dict(region.get("style") or {})
    color = style.pop("color", None)
    if color is not None:
        defaults.update(facecolor=color, edgecolor=color)
    return {**defaults, **style}


def draw_env_on_ax(
    ax,
    env,
    *,
    draw_moving_path=True,
    obs_static_label="Obstacle",
    visit_label="Visit Region",
    moving_obs_label="Moving Obstacle Path",
    moving_obs_snapshots=False,
    x_mask=None,
):
    """Render environment geometry: workspace, goal, obstacles, visit regions, moving paths."""
    env = geometry(env)
    if env.bounds is not None:
        bx, by = env.bounds["x"], env.bounds["y"]
        ax.add_patch(
            patches.Rectangle(
                (bx[0], by[0]),
                bx[1] - bx[0],
                by[1] - by[0],
                **_region_style(env.bounds, facecolor="none",
                                edgecolor=PALETTE["lane"]["stroke"], linestyle="--",
                                linewidth=1.5, zorder=2, label="Workspace"),
            )
        )

    if env.lane_markings:
        draw_road_backdrop(ax, env)
    else:
        if env.goal:
            gx, gy = env.goal["x"], env.goal["y"]
            ax.add_patch(
                patches.Rectangle(
                    (gx[0], gy[0]),
                    gx[1] - gx[0],
                    gy[1] - gy[0],
                    **_region_style(env.goal, facecolor=PALETTE["goal"]["fill"],
                                    edgecolor=PALETTE["goal"]["stroke"],
                                    alpha=0.4, zorder=3, label="Goal"),
                )
            )

    for region in env.visit_regions:
        vx, vy = region["x"], region["y"]
        ax.add_patch(
            patches.Rectangle(
                (vx[0], vy[0]),
                vx[1] - vx[0],
                vy[1] - vy[0],
                facecolor=PALETTE["visit"]["fill"],
                edgecolor=PALETTE["visit"]["stroke"],
                alpha=0.4,
                zorder=3,
                label=visit_label,
            )
        )

    for obs in env.obstacles:
        ox, oy = obs["x"], obs["y"]
        ax.add_patch(
            patches.Rectangle(
                (ox[0], oy[0]),
                ox[1] - ox[0],
                oy[1] - oy[0],
                **_region_style(obs, facecolor=PALETTE["obs_static"]["fill"],
                                edgecolor=PALETTE["obs_static"]["stroke"], alpha=0.6,
                                hatch="//", zorder=4, label=obs_static_label),
            )
        )

    for obs in env.circle_obstacles:
        ax.add_patch(
            patches.Circle(
                obs["center"],
                obs["radius"],
                facecolor=PALETTE["obs_static"]["fill"],
                edgecolor=PALETTE["obs_static"]["stroke"],
                alpha=0.6,
                hatch="//",
                zorder=4,
            )
        )

    if draw_moving_path or moving_obs_snapshots:
        for obs in env.moving_obstacles:
            xt = np.asarray(
                obs["x_traj"].detach().cpu()
                if isinstance(obs["x_traj"], torch.Tensor)
                else obs["x_traj"]
            )
            yt = np.asarray(
                obs["y_traj"].detach().cpu()
                if isinstance(obs["y_traj"], torch.Tensor)
                else obs["y_traj"]
            )
            mask = np.ones(len(xt), dtype=bool)
            if x_mask is not None:
                mask = (xt >= x_mask[0]) & (xt <= x_mask[1])
            if draw_moving_path:
                ax.plot(
                    xt[mask],
                    yt[mask],
                    color=PALETTE["obs_moving"]["stroke"],
                    linestyle="--",
                    alpha=0.4,
                    label=moving_obs_label,
                )
            w, h = obs["width"], obs["height"]
            if moving_obs_snapshots:
                snap_step = max(1, len(xt) // 5)
                for k_i in range(0, len(xt), snap_step):
                    if mask[k_i]:
                        ax.add_patch(
                            patches.Rectangle(
                                (xt[k_i] - w / 2, yt[k_i] - h / 2),
                                w,
                                h,
                                facecolor=PALETTE["obs_moving"]["fill"],
                                edgecolor=PALETTE["obs_moving"]["stroke"],
                                alpha=0.3,
                                zorder=4,
                            )
                        )


def _compute_env_bounds(mean_np, env):
    env = geometry(env)
    x_min, x_max = np.min(mean_np[:, 0]), np.max(mean_np[:, 0])
    y_min, y_max = np.min(mean_np[:, 1]), np.max(mean_np[:, 1])
    for lane in env.lane_markings:
        x_min = min(x_min, min(lane["x"]))
        x_max = max(x_max, max(lane["x"]))
        y_min = min(y_min, lane["y"])
        y_max = max(y_max, lane["y"])
    if env.bounds is not None:
        x_min = min(x_min, env.bounds["x"][0])
        x_max = max(x_max, env.bounds["x"][1])
        y_min = min(y_min, env.bounds["y"][0])
        y_max = max(y_max, env.bounds["y"][1])
    if env.goal:
        x_min = min(x_min, env.goal["x"][0])
        x_max = max(x_max, env.goal["x"][1])
        y_min = min(y_min, env.goal["y"][0])
        y_max = max(y_max, env.goal["y"][1])
    for obs in env.obstacles:
        x_min = min(x_min, obs["x"][0])
        x_max = max(x_max, obs["x"][1])
        y_min = min(y_min, obs["y"][0])
        y_max = max(y_max, obs["y"][1])
    for obs in env.circle_obstacles:
        x_min = min(x_min, obs["center"][0] - obs["radius"])
        x_max = max(x_max, obs["center"][0] + obs["radius"])
        y_min = min(y_min, obs["center"][1] - obs["radius"])
        y_max = max(y_max, obs["center"][1] + obs["radius"])
    for region in env.visit_regions:
        x_min = min(x_min, region["x"][0])
        x_max = max(x_max, region["x"][1])
        y_min = min(y_min, region["y"][0])
        y_max = max(y_max, region["y"][1])
    return x_min, x_max, y_min, y_max


def draw_ego_rect(ax, x, y, heading_deg, rw, rh, alpha, zorder=7):
    t_aff = (
        transforms.Affine2D()
        .translate(-rw / 2, -rh / 2)
        .rotate_deg(heading_deg)
        .translate(x, y)
    )
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            rw,
            rh,
            transform=t_aff + ax.transData,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            linewidth=1.2,
            alpha=alpha,
            zorder=zorder,
        )
    )


def heading_deg(mean_np, t, T):
    dx = mean_np[min(t + 1, T), 0] - mean_np[max(t - 1, 0), 0]
    dy = mean_np[min(t + 1, T), 1] - mean_np[max(t - 1, 0), 1]
    return np.degrees(np.arctan2(dy, dx))


def plot_trajectory(mean_np, cov_np, env, *, show=True, save_path=None):
    env = geometry(env)
    T = mean_np.shape[0] - 1
    x_min, x_max, y_min, y_max = _compute_env_bounds(mean_np, env)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x_min - 1.0, x_max + 1.0)
    ax.set_ylim(y_min - 1.0, y_max + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [m]", fontsize=20, fontweight="bold")
    ax.set_ylabel("$y$ [m]", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)

    draw_env_on_ax(ax, env, moving_obs_snapshots=True)

    if env.goal:
        gx, gy = env.goal["x"], env.goal["y"]
        ax.text(
            (gx[0] + gx[1]) / 2,
            (gy[0] + gy[1]) / 2,
            "G",
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            color=PALETTE["goal"]["stroke"],
            zorder=30,
        )

    for region in env.visit_regions:
        vx, vy = region["x"], region["y"]
        ax.text(
            (vx[0] + vx[1]) / 2,
            (vy[0] + vy[1]) / 2,
            "V",
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            color=PALETTE["visit"]["stroke"],
            zorder=30,
        )

    ax.plot(
        mean_np[:, 0],
        mean_np[:, 1],
        color=PALETTE["ego"]["stroke"],
        linewidth=2.5,
        alpha=0.9,
        label="Trajectory",
        zorder=25,
    )
    for t in range(0, T + 1, 2):
        plot_covariance_ellipse(
            ax,
            mean_np[t, :2],
            cov_np[t, :2, :2],
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.25,
            zorder=15,
            label="Uncertainty" if t == 0 else None,
        )

    start_pos = mean_np[0, :2]
    ax.text(
        start_pos[0] - 0.5,
        start_pos[1],
        "S",
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="center",
        color=PALETTE["ego"]["stroke"],
        zorder=30,
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            ncol=1,
            fontsize=17,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

    _finish(fig, save_path, show)
    return fig, ax


def _to_np(trace):
    """[1, T+1, ...] or [T+1, ...] tensor/array -> numpy without the batch axis."""
    if isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()
    trace = np.asarray(trace)
    return trace[0] if trace.ndim in (3, 4) else trace


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_reach_avoid(
    mean_initial,
    cov_initial,
    mean_final,
    cov_final,
    env,
    ellipse_every=10,
    *,
    show_initial=False,
    title=None,
    save_path=None,
    show=True,
):
    """Predicted stochastic plan; initial beliefs are an optional comparison."""
    env = geometry(env)
    runs = [
        (_to_np(mean_final), _to_np(cov_final), PALETTE["ego"]["stroke"], "-",
         "Predicted belief mean", "95% covariance ellipse"),
    ]
    if show_initial:
        runs.insert(0, (
            _to_np(mean_initial), _to_np(cov_initial), PALETTE["lane"]["stroke"], "--",
            "Initial predicted belief mean", "Initial 95% covariance ellipse",
        ))
    all_means = np.concatenate([mean for mean, *_ in runs], axis=0)
    x_min, x_max, y_min, y_max = _compute_env_bounds(all_means, env)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(x_min - 1.0, x_max + 1.0)
    ax.set_ylim(y_min - 1.0, y_max + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [m]", fontsize=14, fontweight="bold")
    ax.set_ylabel("$y$ [m]", fontsize=14, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)

    draw_env_on_ax(ax, env)
    if env.goal:
        gx, gy = env.goal["x"], env.goal["y"]
        ax.text(
            (gx[0] + gx[1]) / 2, (gy[0] + gy[1]) / 2, "G",
            fontsize=20, fontweight="bold", ha="center", va="center",
            color=PALETTE["goal"]["stroke"], zorder=30,
        )

    for mean, cov, color, style, label, ellipse_label in runs:
        ax.plot(
            mean[:, 0], mean[:, 1], color=color, linestyle=style, linewidth=2.2,
            alpha=0.9, label=label, zorder=25,
        )
        steps = list(range(0, len(mean), ellipse_every))
        for i, t in enumerate(steps):
            plot_covariance_ellipse(
                ax, mean[t, :2], cov[t, :2, :2],
                k=np.sqrt(-2 * np.log(0.05)),  # 95% joint mass in two dimensions
                facecolor=color, edgecolor=color, alpha=0.15, zorder=15,
                label=ellipse_label if i == 0 else None,
            )

    start = runs[0][0][0, :2]
    ax.plot(*start, marker="s", color="k", markersize=8, zorder=31, label="Start")
    ax.text(
        start[0] - 0.5, start[1], "S", fontsize=20, fontweight="bold",
        ha="center", va="center", color="k", zorder=30,
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=9, framealpha=0.95, edgecolor="#cccccc",
    )
    ax.set_title(title or "Predicted belief trajectory", fontsize=13, fontweight="bold")

    _finish(fig, save_path, show)
    return fig, ax


def _altitude_axes(dt, H, threshold, u_max, state_ylim=None):
    """Three panels on one time axis: belief, event probability, controls."""
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )
    ax_state, ax_prob, ax_u = axes
    red, gray = PALETTE["obs_static"]["stroke"], PALETTE["lane"]["stroke"]

    ax_state.axhline(threshold, color=red, linestyle="--", linewidth=1.5,
                     label=f"{threshold:g} m threshold")
    ax_state.set_ylabel("altitude [m]", fontsize=12, fontweight="bold")
    ax_state.set_title(r"Predicted Gaussian belief $N(\mu_k, \sigma_k^2)$",
                       loc="left", fontsize=12, fontweight="bold")
    if state_ylim is not None:
        ax_state.set_ylim(*state_ylim)

    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("probability", fontsize=12, fontweight="bold")
    ax_prob.set_title(rf"$P(Z_k \geq {threshold:g}$ m$)$ and lower score R of "
                      rf"Always$_{{[1,H]}}$", loc="left", fontsize=12, fontweight="bold")

    for bound in (-u_max, u_max):
        ax_u.axhline(bound, color=gray, linestyle=":", linewidth=1.2)
    ax_u.set_ylim(-1.15 * u_max, 1.15 * u_max)
    ax_u.set_ylabel("u [m/s]", fontsize=12, fontweight="bold")
    ax_u.set_xlabel("time [s]", fontsize=12, fontweight="bold")
    ax_u.set_title(rf"Controls $u_k$ (bounds $\pm${u_max:g})", loc="left",
                   fontsize=12, fontweight="bold")

    ax_u.set_xlim(-0.2 * dt, (H + 0.2) * dt)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    return fig, axes


def _draw_altitude_plan(axes, plan, *, dt, name, color, style, alpha=1.0):
    """One plan: +/-2 sigma band and mean, P(Z_k >= threshold), controls."""
    ax_state, ax_prob, ax_u = axes
    mean = _to_np(plan["mean"])[:, 0]
    sigma = np.sqrt(_to_np(plan["cov"])[:, 0, 0])
    time = np.arange(len(mean)) * dt

    ax_state.fill_between(time, mean - 2 * sigma, mean + 2 * sigma, color=color,
                          alpha=0.18 * alpha, label=f"{name} ±2σ band (Gaussian belief)")
    ax_state.plot(time, mean, color=color, linestyle=style, linewidth=2.2, alpha=alpha,
                  label=f"{name} predicted mean")
    ax_prob.plot(time, _to_np(plan["atomic"])[:, 0], color=color, linestyle=style,
                 linewidth=2.2, marker="o", markersize=3, alpha=alpha,
                 label=f"{name}: R = {plan['interval'][0]:.4f}")
    u = _to_np(plan["controls"])[:, 0]  # u_k holds over [t_k, t_k+1)
    ax_u.step(time, np.append(u, u[-1]), where="post", color=color, linestyle=style,
              linewidth=2.2, alpha=alpha, label=f"{name} controls")


def altitude_plan(plan, threshold):
    """Derive altitude plotting data only when a plot/animation requests it."""
    from pdstl.predicates import GreaterThan

    return {
        "mean": plan.rollout.aux["mean_trace"],
        "cov": plan.rollout.aux["cov_trace"],
        "controls": plan.controls,
        "interval": plan.hard_interval,
        "atomic": GreaterThan(threshold, dim=0)(plan.rollout.belief_trajectory)[0],
    }


def plot_altitude_safety(result, *, initial, dt, threshold, u_max, save_path=None, show=True):
    fig, axes = _altitude_axes(dt, len(result.controls), threshold, u_max)
    for plan, name, color, style in (
        (initial, "Initial", PALETTE["lane"]["stroke"], "--"),
        (result, "Optimized", PALETTE["ego"]["stroke"], "-"),
    ):
        _draw_altitude_plan(axes, altitude_plan(plan, threshold), dt=dt,
                            name=name, color=color, style=style)
    for ax in axes:
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _finish(fig, save_path, show)
    return fig, axes


def plot_event_probabilities(dt, traces, *, title=None, save_path=None, show=True):
    """Event probability intervals over time; traces is {label: [T+1, 2]}."""
    colors = cycle((PALETTE["goal"]["stroke"], PALETTE["obs_static"]["stroke"],
                    PALETTE["ego"]["stroke"], PALETTE["visit"]["stroke"],
                    PALETTE["plan"]["stroke"]))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for (label, trace), color in zip(traces.items(), colors):
        trace = _to_np(trace)
        time = np.arange(len(trace)) * dt
        ax.fill_between(time, trace[:, 0], trace[:, 1], color=color, alpha=0.25, label=label)
        ax.plot(time, trace[:, 0], color=color, linewidth=0.9, alpha=0.8)
        ax.plot(time, trace[:, 1], color=color, linewidth=0.9, alpha=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time [s]", fontsize=12, fontweight="bold")
    ax.set_ylabel("probability", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="center right", framealpha=0.95)
    ax.set_title(title or "Event probability intervals", fontsize=12, fontweight="bold")

    _finish(fig, save_path, show)
    return fig, ax


def plot_pdstl_optimization(result, *, save_path=None, show=True):
    """Loss used for optimization; the final hard interval is reporting only."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(result.loss_history, color=PALETTE["ego"]["stroke"], linewidth=2)
    lo, hi = result.hard_interval
    ax.text(0.98, 0.05, f"Final hard pdSTL interval [{lo:.4f}, {hi:.4f}]",
            transform=ax.transAxes, ha="right", va="bottom")
    ax.set(xlabel="Optimization iteration", ylabel="Loss",
           title="Smooth pdSTL objective and control regularization")
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)
    return fig, ax


def visualize_reach_avoid(result, env, *, dt, ellipse_every=10, save_path=None,
                          show=True, initial=None, show_workspace=False):
    """Spatial, event-probability and optimization views of a PlanResult."""
    from functools import reduce
    from pdstl.operators import And
    from planning.environment import reach_avoid_events

    prediction = result.rollout
    mean, cov = prediction.aux["mean_trace"], prediction.aux["cov_trace"]
    spatial = plot_reach_avoid(
        initial.rollout.aux["mean_trace"] if initial else None,
        initial.rollout.aux["cov_trace"] if initial else None,
        mean, cov, env, ellipse_every, show_initial=initial is not None,
        save_path=save_path, show=show,
    )
    events = reach_avoid_events(env)
    trace = prediction.belief_trajectory
    obstacles = events["obstacles"]
    safe = (reduce(And, obstacles)(trace)[0] if obstacles
            else torch.ones(len(trace), 2, device=mean.device))
    traces = {"Goal": events["goal"](trace)[0], "Outside all obstacles": safe}
    if show_workspace:
        traces["Workspace"] = events["workspace"](trace)[0]
    probability_path = optimization_path = None
    if save_path:
        root, ext = os.path.splitext(save_path)
        probability_path = f"{root}_probabilities{ext}"
        optimization_path = f"{root}_optimization{ext}"
    probabilities = plot_event_probabilities(dt, traces, save_path=probability_path, show=show)
    optimization = plot_pdstl_optimization(result, save_path=optimization_path, show=show)
    return spatial, probabilities, optimization


def visualize_mpc(result, env, cfg, *, show, save, output_dir, lane=False):
    """Build presentation traces from executed states and window plans on demand."""
    from pathlib import Path
    from visualization.animation import animate_results

    means = torch.stack([state[0] for state in result.states]).unsqueeze(0)
    covs = torch.stack([state[1] for state in result.states]).unsqueeze(0)
    plans = [plan.rollout.aux["mean_trace"] for plan in result.window_plans]
    scores = [plan.hard_interval[0] for plan in result.window_plans]
    output_dir = Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
    if lane and len(result.applied_controls):
        visualize_lane_change(
            means, covs, result.applied_controls.unsqueeze(0), env,
            p_sat_trace=scores, dt=cfg["dt"], robot_dims=env.metadata.get("robot_dims"),
            xlim=env.metadata.get("plot_xlim"), show=show,
            save_path=str(output_dir / "lane_change.png") if save else None,
        )
    else:
        plot_trajectory(_to_np(means), _to_np(covs), env, show=show,
                        save_path=str(output_dir / cfg.get("figure", "mpc.png")) if save else None)
    if save and cfg.get("animation") and result.window_plans:
        animation = cfg["animation"]
        animate_results(
            means, covs, env, filename=str(output_dir / animation["filename"]),
            step=animation["step"], title=animation["title"], dt=cfg["dt"],
            bounds=animation.get("bounds"), plan_traces=plans,
            robot_dims=cfg.get("robot_dims"),
        )


def plot_controls(u_np):
    T = u_np.shape[0]
    time_steps = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_steps, u_np[:, 0], color=PALETTE["ego"]["stroke"], linewidth=1.8)
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("$u_x$", fontsize=18, fontweight="bold")
    axes[0].tick_params(labelsize=16)
    axes[0].grid(True, alpha=0.35)
    axes[1].plot(time_steps, u_np[:, 1], color=PALETTE["plan"]["stroke"], linewidth=1.8)
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[1].set_ylabel("$u_y$", fontsize=18, fontweight="bold")
    axes[1].set_xlabel("Time Step", fontsize=18, fontweight="bold")
    axes[1].tick_params(labelsize=16)
    axes[1].grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_metrics(history, p_sat_trace):
    if history is None and p_sat_trace is None:
        return
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if p_sat_trace is not None:
        ax.plot(
            p_sat_trace,
            color=PALETTE["goal"]["stroke"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=r"$P_{\downarrow}(\varphi)$",
        )
        ax.set_ylabel(r"$P_{\downarrow}(\varphi)$", fontsize=18, fontweight="bold")
    else:
        ax.plot(history, color=PALETTE["lane"]["stroke"], linewidth=2, label="Loss")
        ax.set_ylabel("Loss", fontsize=18, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=16,
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    fig.subplots_adjust(bottom=0.25)
    plt.show()
    plt.close(fig)


def visualize_results(
    mean_trace,
    cov_trace,
    u_trace,
    env,
    history=None,
    p_sat_trace=None,
    robot_dims=None,
):
    env = geometry(env)
    mean_np = mean_trace.cpu().squeeze().numpy()
    cov_np = cov_trace.cpu().squeeze().numpy()
    u_np = u_trace.cpu().squeeze().numpy()
    plot_trajectory(mean_np, cov_np, env)
    plot_controls(u_np)
    plot_metrics(history, p_sat_trace)


def plot_lc_trajectory(
    ax,
    mean_trace,
    cov_trace,
    env,
    dt,
    robot_dims,
    title=None,
    show_legend=True,
    xlim=None,
):
    env = geometry(env)
    mean_np = mean_trace.cpu().squeeze().numpy()  # [T+1, ≥2]
    cov_np = cov_trace.cpu().squeeze().numpy()  # [T+1, D, D]
    T = mean_np.shape[0] - 1

    road_lo = min(lm["y"] for lm in env.lane_markings) if env.lane_markings else -2.0
    road_hi = max(lm["y"] for lm in env.lane_markings) if env.lane_markings else 6.0
    x_lo = mean_np[:, 0].min() - 1.5
    x_hi = mean_np[:, 0].max() + 1.5
    y_lo, y_hi = road_lo - 1.2, road_hi + 1.2

    if xlim:
        ax.set_xlim(xlim)
        x_lo = min(x_lo, xlim[0])
        x_hi = max(x_hi, xlim[1])
    else:
        ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("$y$ [m]", fontsize=24)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title, fontsize=22, fontweight="bold")

    draw_env_on_ax(
        ax,
        env,
        obs_static_label="Stopped Vehicle",
        visit_label="Merge Zone",
        moving_obs_label="Other Vehicle Path",
        moving_obs_snapshots=True,
        x_mask=(x_lo, x_hi),
    )

    # Uncertainty tube
    step_ell = max(1, T // 16)
    for t in range(0, T + 1, step_ell):
        plot_covariance_ellipse(
            ax,
            mean_np[t, :2],
            cov_np[t, :2, :2],
            k=2.45,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.16,
            zorder=5,
            label="95% CI" if t == 0 else None,
        )

    ax.plot(
        mean_np[:, 0],
        mean_np[:, 1],
        color=PALETTE["ego"]["stroke"],
        linewidth=2.2,
        alpha=0.9,
        zorder=7,
        label="Ego Trajectory",
    )
    if robot_dims:
        rw, rh = robot_dims
        for t in range(0, T + 1, max(1, T // 10)):
            draw_ego_rect(
                ax,
                mean_np[t, 0],
                mean_np[t, 1],
                heading_deg(mean_np, t, T),
                rw,
                rh,
                alpha=0.30,
                zorder=6,
            )

    ax.plot(
        mean_np[0, 0],
        mean_np[0, 1],
        "o",
        color=PALETTE["ego"]["stroke"],
        markersize=6,
        zorder=9,
    )
    ax.plot(
        mean_np[-1, 0],
        mean_np[-1, 1],
        "s",
        color=PALETTE["ego"]["stroke"],
        markersize=6,
        zorder=9,
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    if show_legend:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(by_label),
            fontsize=18,
            framealpha=0.95,
        )

    return by_label


def plot_lc_snapshots(
    ax,
    mean_trace,
    cov_trace,
    env,
    dt,
    robot_dims,
    title=None,
    show_legend=True,
    show_xlabel=True,
    xlim=None,
):
    env = geometry(env)
    mean_np = mean_trace.cpu().squeeze().numpy()  # [T+1, ≥2]
    cov_np = cov_trace.cpu().squeeze().numpy()  # [T+1, D, D]
    T = mean_np.shape[0] - 1

    road_lo = min(lm["y"] for lm in env.lane_markings) if env.lane_markings else -2.0
    road_hi = max(lm["y"] for lm in env.lane_markings) if env.lane_markings else 6.0
    x_lo = mean_np[:, 0].min() - 1.5
    x_hi = mean_np[:, 0].max() + 1.5
    y_lo, y_hi = road_lo - 1.2, road_hi + 1.2

    N_SNAP = 6
    snap_t = np.linspace(0, T, N_SNAP, dtype=int)

    if xlim:
        ax.set_xlim(xlim)
        x_lo = min(x_lo, xlim[0])
        x_hi = max(x_hi, xlim[1])
    else:
        ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    if show_xlabel:
        ax.set_xlabel("$x$ [m]", fontsize=24)
    ax.set_ylabel("$y$ [m]", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title, fontsize=22, fontweight="bold")

    draw_env_on_ax(ax, env, x_mask=(x_lo, x_hi))

    # Thin full trajectory as reference line
    ax.plot(
        mean_np[:, 0],
        mean_np[:, 1],
        color=PALETTE["ego"]["stroke"],
        linewidth=1.2,
        alpha=0.35,
        zorder=4,
    )

    # Legend patches
    ego_patch = patches.Patch(
        facecolor=PALETTE["ego"]["fill"],
        edgecolor=PALETTE["ego"]["stroke"],
        linewidth=1.2,
        label="Ego Vehicle",
    )
    obs_patch = patches.Patch(
        facecolor=PALETTE["obs_moving"]["fill"],
        edgecolor=PALETTE["obs_moving"]["stroke"],
        linewidth=1.2,
        label="Other Vehicle",
    )
    ci_patch = patches.Patch(
        facecolor=PALETTE["ego"]["fill"],
        edgecolor=PALETTE["ego"]["stroke"],
        linewidth=0.8,
        alpha=0.25,
        label="95% CI",
    )
    goal_patch = patches.Patch(
        facecolor=PALETTE["goal"]["fill"],
        edgecolor="none",
        alpha=0.5,
        label="Goal Lane",
    )
    legend_handles = [ego_patch, obs_patch]
    if env.obstacles:
        static_patch = patches.Patch(
            facecolor=PALETTE["obs_static"]["fill"],
            edgecolor=PALETTE["obs_static"]["stroke"],
            linewidth=1.2,
            label="Stopped Vehicle",
        )
        legend_handles.append(static_patch)
    if env.visit_regions:
        visit_patch = patches.Patch(
            facecolor=PALETTE["visit"]["fill"],
            edgecolor=PALETTE["visit"]["stroke"],
            alpha=0.5,
            label="Merge Zone",
        )
        legend_handles.append(visit_patch)
    legend_handles.extend([ci_patch, goal_patch])

    # Faint intermediate covariance tube
    for t in range(0, T + 1, max(1, T // 20)):
        plot_covariance_ellipse(
            ax,
            mean_np[t, :2],
            cov_np[t, :2, :2],
            k=2.45,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.06,
            zorder=4,
        )

    for ki, t in enumerate(snap_t):
        frac = ki / (N_SNAP - 1)
        alpha = 0.35 + 0.55 * frac
        t_sec = t * dt

        plot_covariance_ellipse(
            ax,
            mean_np[t, :2],
            cov_np[t, :2, :2],
            k=2.45,
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.10 + 0.15 * frac,
            zorder=5,
        )

        ex, ey = mean_np[t, 0], mean_np[t, 1]
        if robot_dims:
            rw, rh = robot_dims
            draw_ego_rect(ax, ex, ey, heading_deg(mean_np, t, T), rw, rh, alpha=alpha, zorder=7)
            label_y_off = rh / 2 + 0.45
        else:
            ax.plot(
                ex,
                ey,
                "o",
                color=PALETTE["ego"]["stroke"],
                markersize=6,
                alpha=alpha,
                zorder=7,
            )
            label_y_off = 0.5

        ax.annotate(
            f"$t={t_sec:.1f}\\,$s",
            xy=(ex, ey + label_y_off),
            fontsize=12,
            ha="center",
            va="bottom",
            color=PALETTE["ego"]["stroke"],
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

        for obs in env.moving_obstacles:
            xt = np.asarray(
                obs["x_traj"].detach().cpu()
                if isinstance(obs["x_traj"], torch.Tensor)
                else obs["x_traj"]
            )
            yt = np.asarray(
                obs["y_traj"].detach().cpu()
                if isinstance(obs["y_traj"], torch.Tensor)
                else obs["y_traj"]
            )
            if t < len(xt) and x_lo <= xt[t] <= x_hi:
                ax.add_patch(
                    patches.Rectangle(
                        (xt[t] - obs["width"] / 2, yt[t] - obs["height"] / 2),
                        obs["width"],
                        obs["height"],
                        facecolor=PALETTE["obs_moving"]["fill"],
                        edgecolor=PALETTE["obs_moving"]["stroke"],
                        linewidth=1.0,
                        alpha=alpha,
                        zorder=6,
                    )
                )

    if show_legend:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_handles),
            fontsize=22,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

    return legend_handles


def visualize_lane_change(
    mean_trace,
    cov_trace,
    u_trace,
    env,
    p_sat_trace=None,
    dt=0.2,
    robot_dims=None,
    xlim=None,
    show=True,
    save_path=None,
):
    env = geometry(env)
    mean_np = mean_trace.cpu().squeeze().numpy()  # [T+1, ≥2]
    u_np = u_trace.detach().cpu().reshape(-1, u_trace.shape[-1]).numpy()  # [T,  2]
    T = mean_np.shape[0] - 1
    time_u = np.arange(T) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    h_traj = plot_lc_trajectory(
        ax1, mean_trace, cov_trace, env, dt, robot_dims, show_legend=False, xlim=xlim
    )
    h_snap = plot_lc_snapshots(
        ax2, mean_trace, cov_trace, env, dt, robot_dims, show_legend=False, xlim=xlim
    )

    combined_handles = []
    seen = set()
    for lbl, h in h_traj.items():
        if lbl not in seen:
            seen.add(lbl)
            combined_handles.append(h)
    for h in h_snap:
        lbl = h.get_label()
        if lbl not in seen:
            seen.add(lbl)
            combined_handles.append(h)

    fig.legend(
        handles=combined_handles,
        loc="lower center",
        ncol=min(len(combined_handles), 4),
        fontsize=16,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    _finish(fig, save_path, show)

    n_rows = 3 if p_sat_trace is not None else 2
    fig3, axes = plt.subplots(n_rows, 1, figsize=(10, 2.6 * n_rows), sharex=True)

    axes[0].plot(time_u, u_np[:, 0], color=PALETTE["ego"]["stroke"], linewidth=1.8)
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("$a_x$ [m/s²]", fontsize=16)
    axes[0].set_title(
        "Control Inputs and Satisfaction Probability", fontsize=18, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.35)
    axes[0].tick_params(labelsize=14)

    axes[1].plot(time_u, u_np[:, 1], color=PALETTE["plan"]["stroke"], linewidth=1.8)
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[1].set_ylabel("$a_y$ [m/s²]", fontsize=16)
    axes[1].grid(True, alpha=0.35)
    axes[1].tick_params(labelsize=14)

    if p_sat_trace is not None:
        p_sat_arr = np.asarray(p_sat_trace)
        axes[2].plot(
            time_u[: len(p_sat_arr)],
            p_sat_arr,
            color=PALETTE["goal"]["stroke"],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=r"$P_{\downarrow}(\varphi)$",
        )
        axes[2].axhline(
            0.85,
            color="k",
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
            label="Threshold ($\\alpha = 0.85$)",
        )
        axes[2].set_ylim(0, 1.05)
        axes[2].set_ylabel(r"$P_{\downarrow}(\varphi)$", fontsize=16)
        axes[2].legend(fontsize=14, loc="lower right", framealpha=0.9)
        axes[2].grid(True, alpha=0.35)
        axes[2].tick_params(labelsize=14)

    axes[-1].set_xlabel("Time [s]", fontsize=16)
    plt.tight_layout()
    metrics_path = None if save_path is None else os.path.splitext(save_path)[0] + "_metrics.png"
    _finish(fig3, metrics_path, show)
