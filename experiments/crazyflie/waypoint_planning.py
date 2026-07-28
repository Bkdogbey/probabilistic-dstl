"""Offline pdSTL planning for the Crazyflie experiment.

Optimises a waypoint plan for one fan level/scenario and writes it to
waypoints/pdstl[_<scenario>]_fan<L>.json. Called by `run.py plan`; has no
hardware/ROS dependency (only torch/numpy, plus matplotlib for --plot).

Shared CLI runner + plotting code for all three scenarios -- environment/
planner/dynamics construction is dimension-specific and lives in
components/planning_2d.py / components/planning_3d.py. `run_plan` dispatches
to `_run_plan_2d`/`_run_plan_3d`/`_run_plan_figure8`, which share
`_optimize_and_report` but plot with dimension-specific code: `_plot_2d`/
`_draw_env_2d` for the baseline, `_plot`/`_draw_env` for gate and figure8
(figure8 passes its own obstacle list and no reference-line curve).
"""

from __future__ import annotations

import datetime

import numpy as np
import torch

# components.config must import before planning.*/pdstl.* below -- it does
# the sys.path insert those packages need.
from components.config import (
    DT,
    EXPERIMENT_DIR,
    FIG8_FLIGHT_POINTS,
    FIG8_OBSTACLES,
    FIG8_PLOT_POINTS,
    FIG8_RETURN_TOLERANCE,
    FLIGHT_Z_BOUNDS,
    OBSTACLES,
    PLANNER_ALPHA,
    SAFE_PATH_FLIGHT_POINTS,
    Q_STD,
    SIGMA0_PER_FAN,
    Z_HEIGHT,
    FLIGHT_X_BOUNDS,
    FLIGHT_Y_BOUNDS,
    obstacle_clearance_2d,
    reference_direct_path,
    save_pdstl_waypoints,
    validate_waypoints_in_bounds,
)
from pdstl.base import BeliefTrajectory
from planning.planner import TorchGaussianBelief

from components.planning_2d import (
    build_planner_2d,
    nominal_safe_waypoints,
    x0_belief_2d,
    _nominal_init_guess_2d,
)
from components.planning_3d import (
    build_planner_3d,
    build_planner_figure8,
    figure8_min_clearances,
    nominal_figure8_waypoints,
    nominal_gate_waypoints,
    terminal_return_error,
    x0_belief_3d,
    x0_belief_figure8,
    _nominal_init_guess_3d,
    _nominal_init_guess_figure8,
)


def _evaluate_rho(planner, init_u: torch.Tensor,
                  x0_mean: torch.Tensor, x0_cov: torch.Tensor) -> float:
    """Evaluate STL P(sat) for a given control sequence without optimising."""
    u_norm = torch.clamp(init_u / planner.dyn.u_max, -0.99, 0.99)
    v = 0.5 * torch.log((1 + u_norm) / (1 - u_norm))
    with torch.no_grad():
        mean_trace, cov_trace = planner.dyn(v, x0_mean, x0_cov)
        beliefs = [TorchGaussianBelief(mean_trace[:, t, :], cov_trace[:, t]) for t in range(planner.T + 1)]
        phi = planner.env.get_specification(planner.T)
        return phi(BeliefTrajectory(beliefs))[0, 0, 0].item()


def _print_rho(label: str, value: float) -> None:
    print(f'{label + ":":<48} {value:.4f}')


def _optimize_and_report(planner, x0_mean: torch.Tensor, x0_cov: torch.Tensor,
                         init_u: torch.Tensor):
    """Evaluate rho_before, run a single optimisation from init_u, print both.

    Returns (rho_before, best_mean, best_cov, best_u, best_p).
    """
    rho_before = _evaluate_rho(planner, init_u, x0_mean, x0_cov)
    _print_rho('rho_before (deterministic nominal path)', rho_before)

    best_mean, best_cov, best_u, best_p, _history = planner._optimize_window(
        x0_mean, x0_cov, init_guess=init_u, verbose=True,
    )
    _print_rho('rho_after (optimised)', best_p)

    return rho_before, best_mean, best_cov, best_u, best_p


def _axis_clearance(pos: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per-point distance outside [lo, hi] along one axis (0 where pos is inside).

    Shared building block for both the 2D and 3D obstacle-clearance
    functions below -- they differ only in how many axes they combine.
    """
    return np.maximum(np.maximum(lo - pos, 0.0), pos - hi)


# ── 2D plotting (baseline scenario only) ────────────────────────────────────
# obstacle_clearance_2d (min x,y-only distance from a curve to an obstacle
# box) is imported from components.config -- shared with the closed-form
# sine-amplitude calculation in planning_2d.py, not redefined here.


def _draw_env_2d(ax, env) -> None:
    """Draw the arena (bounds, obstacle boxes, goal box) as flat 2D rectangles.

    Used only for the baseline scenario -- planning_2d.Environment carries no
    z on any region, so there's no altitude axis to draw at all, unlike the
    gate scenario's 3D _draw_env below.
    """
    import matplotlib.patches as patches

    ax.set_xlim(FLIGHT_X_BOUNDS[0] - 0.1, FLIGHT_X_BOUNDS[1] + 0.1)
    ax.set_ylim(FLIGHT_Y_BOUNDS[0] - 0.1, FLIGHT_Y_BOUNDS[1] + 0.1)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.3)

    ax.add_patch(patches.Rectangle(
        (FLIGHT_X_BOUNDS[0], FLIGHT_Y_BOUNDS[0]),
        FLIGHT_X_BOUNDS[1] - FLIGHT_X_BOUNDS[0], FLIGHT_Y_BOUNDS[1] - FLIGHT_Y_BOUNDS[0],
        facecolor='none', edgecolor='black', linestyle='dashed', alpha=0.4,
    ))
    for obs in env.obstacles:
        ox, oy = obs['x'], obs['y']
        ax.add_patch(patches.Rectangle(
            (ox[0], oy[0]), ox[1] - ox[0], oy[1] - oy[0],
            facecolor='red', edgecolor='darkred', alpha=0.35,
        ))
    if env.goal:
        gx, gy = env.goal['x'], env.goal['y']
        ax.add_patch(patches.Rectangle(
            (gx[0], gy[0]), gx[1] - gx[0], gy[1] - gy[0],
            facecolor='green', edgecolor='darkgreen', alpha=0.25,
        ))


def _plot_2d(env, fan, sigma0, q_std, nominal_wps, nominal_curve, reference_curve,
             opt_xy, opt_cov, out_path) -> None:
    """2D before/after comparison plot for the baseline mission.

    Flat axes and Ellipse covariance patches -- matches the arena's actual
    planning dimensionality (no z), unlike _plot's 3D-axes/ellipsoid version
    below (gate scenario only).
    """
    import math

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f'Fan {fan}  (2D baseline, Σ0 = {sigma0} m², q_std = {q_std} m/step)',
        fontsize=13,
    )

    # ── Before: deterministic safe path, obstacle-free reference + clearances ──
    _draw_env_2d(ax_before, env)
    ax_before.set_title('Before (Deterministic Safe Path)')
    ax_before.plot(reference_curve[:, 0], reference_curve[:, 1],
                   'k--', lw=1.2, alpha=0.6, label='reference (no obstacles)')
    ax_before.plot(nominal_curve[:, 0], nominal_curve[:, 1], 'b-', lw=1.5, alpha=0.8)
    ax_before.scatter(nominal_wps[:, 0], nominal_wps[:, 1], c='blue', s=25, label='flown waypoints')
    ax_before.scatter(*nominal_wps[0], c='green', s=80, label='start')
    ax_before.scatter(*nominal_wps[-1], c='red', s=80, marker='s', label='end')
    for obs in OBSTACLES:
        clearance = obstacle_clearance_2d(nominal_curve, obs)
        cx = (obs['x'][0] + obs['x'][1]) / 2
        cy = (obs['y'][0] + obs['y'][1]) / 2
        ax_before.text(cx, cy, f"{obs['name']}\n{clearance * 100:.1f} cm",
                       ha='center', va='center', fontsize=7, color='darkred')
    ax_before.legend(fontsize=8)

    # ── After: pdSTL-optimised plan, with this fan's belief-covariance ellipses ──
    _draw_env_2d(ax_after, env)
    ax_after.set_title('After (pdSTL Optimised)')
    ax_after.plot(reference_curve[:, 0], reference_curve[:, 1], 'k--', lw=1.2, alpha=0.6)
    ax_after.plot(opt_xy[:, 0], opt_xy[:, 1], 'b.-', lw=2, ms=8)
    ax_after.scatter(*opt_xy[0], c='green', s=80, label='start')
    ax_after.scatter(*opt_xy[-1], c='red', s=80, marker='s', label='end')
    for obs in OBSTACLES:
        clearance = obstacle_clearance_2d(opt_xy, obs)
        cx = (obs['x'][0] + obs['x'][1]) / 2
        cy = (obs['y'][0] + obs['y'][1]) / 2
        ax_after.text(cx, cy, f"{obs['name']}\n{clearance * 100:.1f} cm",
                      ha='center', va='center', fontsize=7, color='darkred')
    for t in range(len(opt_xy)):
        vals, vecs = np.linalg.eigh(opt_cov[t])
        angle = math.degrees(math.atan2(*vecs[:, -1][::-1]))
        w, h = 2 * 2 * np.sqrt(np.maximum(vals, 0.0))
        ax_after.add_patch(Ellipse(
            xy=opt_xy[t], width=w, height=h, angle=angle,
            edgecolor='blue', facecolor='none', alpha=0.5, lw=0.8,
        ))
    ax_after.legend(fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f'Plot saved to {out_path}')
    plt.show()
    plt.close(fig)


# ── 3D plotting (gate scenario only) ────────────────────────────────────────
def _obstacle_clearance(curve_xyz: np.ndarray, obs: dict) -> float:
    """Min distance from any point on curve_xyz [N,3] to an {'x','y','z'} obstacle box (0 = inside).

    Correctly reads as positive when flown over the top of a box (large z,
    dz > 0) even with zero lateral (x,y) clearance, not just when routed
    around it.
    """
    z0, z1 = obs.get('z', (curve_xyz[:, 2].min(), curve_xyz[:, 2].max()))
    dx = _axis_clearance(curve_xyz[:, 0], *obs['x'])
    dy = _axis_clearance(curve_xyz[:, 1], *obs['y'])
    dz = _axis_clearance(curve_xyz[:, 2], z0, z1)
    return float(np.min(np.sqrt(dx**2 + dy**2 + dz**2)))


def _box_faces(x_range, y_range, z_range) -> list[np.ndarray]:
    """Return the 6 quad faces of an axis-aligned box as arrays of 4 corner points."""
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    c = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    idx = [
        [0, 1, 2, 3], [4, 5, 6, 7],  # bottom, top
        [0, 1, 5, 4], [2, 3, 7, 6],  # front, back
        [1, 2, 6, 5], [0, 3, 7, 4],  # right, left
    ]
    return [c[i] for i in idx]


def _draw_box(ax, x_range, y_range, z_range, *, facecolor, edgecolor, alpha, linestyle='solid') -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = _box_faces(x_range, y_range, z_range)
    ax.add_collection3d(Poly3DCollection(
        faces, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle,
    ))


def _draw_env(ax, env) -> None:
    """Draw the arena (workspace bounds wireframe, obstacle boxes, goal box) in 3D.

    `env` is duck-typed -- works for both the 2D planning_2d.Environment (via
    build_environment_2d, no z on any region) and the 3D planning_3d.Environment3D
    (via build_environment_3d, z on every region) -- .get('z', ...) covers the
    former, getattr(..., 'time_windowed_bounds', []) covers the fact that only
    Environment3D defines that attribute at all.
    """
    # Use the env's own bounds (if set) rather than always the global arena
    # bounds -- gate's workspace equals the arena bounds so this is a no-op
    # for it, but figure8 has its own narrower workspace and the wireframe
    # box/axis limits should reflect that, not the arena.
    env_bounds = getattr(env, 'bounds', None) or {}
    x_bounds = env_bounds.get('x', FLIGHT_X_BOUNDS)
    y_bounds = env_bounds.get('y', FLIGHT_Y_BOUNDS)
    z_bounds = env_bounds.get('z', FLIGHT_Z_BOUNDS)

    ax.set_xlim(x_bounds[0] - 0.1, x_bounds[1] + 0.1)
    ax.set_ylim(y_bounds[0] - 0.1, y_bounds[1] + 0.1)
    ax.set_zlim(z_bounds[0] - 0.1, z_bounds[1] + 0.1)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    # facecolor=(0,0,0,0) (fully transparent RGBA), not the string 'none' --
    # this matplotlib/mpl_toolkits version's Poly3DCollection.do_3d_projection
    # crashes (ValueError: not enough values to unpack) on a genuinely
    # colorless face; a transparent RGBA renders identically (invisible face,
    # visible edges) without hitting that code path.
    _draw_box(
        ax, x_bounds, y_bounds, z_bounds,
        facecolor=(0, 0, 0, 0), edgecolor='black', alpha=0.15, linestyle='dashed',
    )
    for obs in env.obstacles:
        _draw_box(
            ax, obs['x'], obs['y'], obs.get('z', FLIGHT_Z_BOUNDS),
            facecolor='red', edgecolor='darkred', alpha=0.35,
        )
    if env.goal:
        _draw_box(
            ax, env.goal['x'], env.goal['y'], env.goal.get('z', FLIGHT_Z_BOUNDS),
            facecolor='green', edgecolor='darkgreen', alpha=0.25,
        )
    # Gate scenario only: the 2D env's timed_visit_regions is always empty and
    # it has no time_windowed_bounds attribute at all, so these loops draw
    # nothing for the 2D baseline.
    for region in env.timed_visit_regions:
        _draw_box(
            ax, region['x'], region['y'], region.get('z', FLIGHT_Z_BOUNDS),
            facecolor='blue', edgecolor='darkblue', alpha=0.35,
        )
    for region in getattr(env, 'time_windowed_bounds', []):
        _draw_box(
            ax, region['x'], region['y'], region['z'],
            facecolor=(0, 0, 0, 0), edgecolor='purple', alpha=0.5, linestyle='dashed',
        )


def _cov_ellipsoid_surface(center: np.ndarray, cov: np.ndarray, n_std: float = 2.0, resolution: int = 8):
    """Parametric 2σ ellipsoid surface (X, Y, Z grids) from a 3x3 covariance, for plot_surface."""
    vals, vecs = np.linalg.eigh(cov)
    radii = n_std * np.sqrt(np.maximum(vals, 0.0))

    u = np.linspace(0.0, 2 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    sphere = np.stack([
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
    ], axis=-1)  # [res, res, 3]

    ellipsoid = (sphere * radii) @ vecs.T + center
    return ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]


def _plot(env, fan, sigma0, q_std, nominal_wps, nominal_curve, reference_curve,
          opt_xyz, opt_cov, out_path, *, obstacles: list[dict] | None = None) -> None:
    """reference_curve may be None to skip the dashed 'no obstacles' reference
    line (no meaningful straight-line reference exists for a closed loop, so
    the figure8 scenario passes None). obstacles defaults to the module-level
    OBSTACLES global (gate/baseline's arena set); figure8 passes its own
    FIG8_OBSTACLES so the clearance annotations match whichever env was
    actually built, instead of always annotating against the arena set.
    """
    import matplotlib.pyplot as plt

    obs_list = OBSTACLES if obstacles is None else obstacles

    fig = plt.figure(figsize=(15, 7.5))
    ax_before = fig.add_subplot(1, 2, 1, projection='3d')
    ax_after = fig.add_subplot(1, 2, 2, projection='3d')
    fig.suptitle(
        f'Fan {fan}  (Σ0 = {sigma0} m², q_std = {q_std} m/step)', fontsize=13,
    )

    # ── Before: deterministic safe path, obstacle-free reference + clearances ──
    _draw_env(ax_before, env)
    ax_before.set_title('Before (Deterministic Safe Path)')
    if reference_curve is not None:
        ax_before.plot(reference_curve[:, 0], reference_curve[:, 1], reference_curve[:, 2],
                       'k--', lw=1.2, alpha=0.6, label='reference (no obstacles)')
    ax_before.plot(nominal_curve[:, 0], nominal_curve[:, 1], nominal_curve[:, 2],
                   'b-', lw=1.5, alpha=0.8)
    ax_before.scatter(nominal_wps[:, 0], nominal_wps[:, 1], nominal_wps[:, 2],
                      c='blue', s=25, label='flown waypoints')
    ax_before.scatter(*nominal_wps[0], c='green', s=80, label='start')
    ax_before.scatter(*nominal_wps[-1], c='red', s=80, marker='s', label='end')
    for obs in obs_list:
        clearance = _obstacle_clearance(nominal_curve, obs)
        cx = (obs['x'][0] + obs['x'][1]) / 2
        cy = (obs['y'][0] + obs['y'][1]) / 2
        cz = (obs.get('z', (0.0, obs.get('height', 0.0)))[1]) / 2
        ax_before.text(cx, cy, cz, f"{obs['name']}\n{clearance * 100:.1f} cm",
                      ha='center', va='center', fontsize=7, color='darkred')
    ax_before.legend(fontsize=8)

    # ── After: pdSTL-optimised plan, with this fan's belief-covariance ellipsoids ──
    _draw_env(ax_after, env)
    ax_after.set_title('After (pdSTL Optimised)')
    if reference_curve is not None:
        ax_after.plot(reference_curve[:, 0], reference_curve[:, 1], reference_curve[:, 2],
                      'k--', lw=1.2, alpha=0.6)
    ax_after.plot(opt_xyz[:, 0], opt_xyz[:, 1], opt_xyz[:, 2], 'b.-', lw=2, ms=8)
    ax_after.scatter(*opt_xyz[0], c='green', s=80, label='start')
    ax_after.scatter(*opt_xyz[-1], c='red', s=80, marker='s', label='end')
    for obs in obs_list:
        clearance = _obstacle_clearance(opt_xyz, obs)
        cx = (obs['x'][0] + obs['x'][1]) / 2
        cy = (obs['y'][0] + obs['y'][1]) / 2
        cz = (obs.get('z', (0.0, obs.get('height', 0.0)))[1]) / 2
        ax_after.text(cx, cy, cz, f"{obs['name']}\n{clearance * 100:.1f} cm",
                     ha='center', va='center', fontsize=7, color='darkred')
    for t in range(len(opt_xyz)):
        X, Y, Z = _cov_ellipsoid_surface(opt_xyz[t], opt_cov[t])
        ax_after.plot_surface(X, Y, Z, color='blue', alpha=0.12, linewidth=0)
    ax_after.legend(fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f'Plot saved to {out_path}')
    plt.show()
    plt.close(fig)


# ── Entry ────────────────────────────────────────────────────────────────────
def _save_plan(
    fan: int, scenario: str, sigma0: float, q_std: float,
    rho_before: float, best_p: float,
    waypoints: list[tuple[float, float, float]], *,
    T: int, dt: float, uncertainty_source: str = 'sigma0_per_fan_table',
    extra_meta: dict | None = None,
) -> None:
    """Validate + write the optimised plan JSON. Shared by all three scenario drivers."""
    validate_waypoints_in_bounds(waypoints, label='Generated waypoint')
    meta = {
        'sigma0': sigma0,
        'q_std': q_std,
        'rho_before': round(rho_before, 4),
        'rho_after': round(float(best_p), 4),
        'alpha': PLANNER_ALPHA,
        'T': T,
        'dt': dt,
        'uncertainty_source': uncertainty_source,
        'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        **(extra_meta or {}),
    }
    out = save_pdstl_waypoints(fan, waypoints, meta, scenario=scenario)
    print(f'Wrote {len(waypoints)} waypoints to {out}')


def _run_plan_2d(fan: int, plot: bool) -> None:
    """Optimise and save a 2D baseline plan. Optimizer state is strictly (x, y)."""
    sigma0 = SIGMA0_PER_FAN[fan]
    q_std = Q_STD
    planner, _dynamics, env = build_planner_2d(fan)
    print(f'Planning for fan {fan}  (Σ0 = {sigma0} m², q_std = {q_std} m/step, '
          f'T = {planner.T}, scenario = baseline [2D])')

    x0_mean, x0_cov = x0_belief_2d(fan)
    init_u = _nominal_init_guess_2d()
    rho_before, best_mean, best_cov, _best_u, best_p = _optimize_and_report(
        planner, x0_mean, x0_cov, init_u,
    )

    positions_xy = best_mean.squeeze(0).cpu().numpy()
    assert positions_xy.shape[-1] == 2, f'2D planned positions must be 2D, got {positions_xy.shape}'
    waypoints = [(float(x), float(y), Z_HEIGHT) for x, y in positions_xy]
    _save_plan(fan, 'baseline', sigma0, q_std, rho_before, best_p, waypoints, T=planner.T, dt=DT)

    if plot:
        nominal_wps = np.array(nominal_safe_waypoints(n_points=SAFE_PATH_FLIGHT_POINTS))[:, :2]
        nominal_curve = np.array(nominal_safe_waypoints(n_points=200))[:, :2]
        reference_curve = np.array(reference_direct_path(n_points=200))
        plot_path = EXPERIMENT_DIR / 'plots' / f'fan{fan}_comparison.png'
        opt_cov = best_cov.squeeze(0).cpu().numpy()
        assert opt_cov.shape[-2:] == (2, 2), f'2D planned covariance must be 2x2, got {opt_cov.shape}'

        _plot_2d(env, fan, sigma0, q_std, nominal_wps, nominal_curve, reference_curve,
                 positions_xy, opt_cov, plot_path)


def _run_plan_3d(fan: int, plot: bool) -> None:
    """Optimise and save a 3D gate plan. Optimizer state is (x, y, z)."""
    sigma0 = SIGMA0_PER_FAN[fan]
    q_std = Q_STD
    planner, _dynamics, env = build_planner_3d()
    print(f'Planning for fan {fan}  (Σ0 = {sigma0} m², q_std = {q_std} m/step, '
          f'T = {planner.T}, scenario = gate [3D])')

    x0_mean, x0_cov = x0_belief_3d(fan)
    init_u = _nominal_init_guess_3d()
    rho_before, best_mean, best_cov, _best_u, best_p = _optimize_and_report(
        planner, x0_mean, x0_cov, init_u,
    )

    positions_xyz = best_mean.squeeze(0).cpu().numpy()
    assert positions_xyz.shape[-1] == 3, f'3D planned positions must be 3D, got {positions_xyz.shape}'
    waypoints = [(float(x), float(y), float(z)) for x, y, z in positions_xyz]
    _save_plan(fan, 'gate', sigma0, q_std, rho_before, best_p, waypoints, T=planner.T, dt=DT)

    if plot:
        nominal_wps = np.array(nominal_gate_waypoints(n_points=SAFE_PATH_FLIGHT_POINTS))
        nominal_curve = np.array(nominal_gate_waypoints(n_points=200))
        reference_xy = np.array(reference_direct_path(n_points=200))
        reference_curve = np.column_stack([reference_xy, np.full(len(reference_xy), Z_HEIGHT)])
        plot_path = EXPERIMENT_DIR / 'plots' / f'fan{fan}_gate_comparison.png'
        opt_cov = best_cov.squeeze(0).cpu().numpy()
        assert opt_cov.shape[-2:] == (3, 3), f'planned covariance must be 3x3, got {opt_cov.shape}'
        _plot(env, fan, sigma0, q_std, nominal_wps, nominal_curve, reference_curve,
              positions_xyz, opt_cov, plot_path)


def _run_plan_figure8(fan: int, plot: bool) -> None:
    """Optimise and save a figure8 plan. Optimizer state is (x, y, z)."""
    sigma0 = SIGMA0_PER_FAN[fan]
    q_std = Q_STD
    planner, _dynamics, env = build_planner_figure8()
    print(f'Planning for fan {fan}  (Σ0 = {sigma0} m², q_std = {q_std} m/step, '
          f'T = {planner.T}, scenario = figure8 [3D])')

    x0_mean, x0_cov = x0_belief_figure8(fan)
    init_u = _nominal_init_guess_figure8()
    rho_before, best_mean, best_cov, _best_u, best_p = _optimize_and_report(
        planner, x0_mean, x0_cov, init_u,
    )

    positions_xyz = best_mean.squeeze(0).cpu().numpy()
    assert positions_xyz.shape[-1] == 3, f'figure8 planned positions must be 3D, got {positions_xyz.shape}'
    waypoints = [(float(x), float(y), float(z)) for x, y, z in positions_xyz]

    return_error = terminal_return_error(waypoints)
    print(f'{"terminal return error (||mu[T]-mu[0]||):":<48} '
          f'{return_error:.4f} m (tolerance {FIG8_RETURN_TOLERANCE} m)')

    _save_plan(
        fan, 'figure8', sigma0, q_std, rho_before, best_p, waypoints,
        T=planner.T, dt=DT, uncertainty_source='sigma0_per_fan_table',
        extra_meta={
            'return_tolerance': FIG8_RETURN_TOLERANCE,
            'return_error': round(return_error, 4),
        },
    )

    if plot:
        nominal_wps = np.array(nominal_figure8_waypoints(n_points=FIG8_FLIGHT_POINTS))
        nominal_curve = np.array(nominal_figure8_waypoints(n_points=FIG8_PLOT_POINTS))
        print('Dense-curve minimum clearance per obstacle:')
        for name, clearance in figure8_min_clearances(nominal_curve).items():
            print(f'  {name}: {clearance * 100:.1f} cm')
        plot_path = EXPERIMENT_DIR / 'plots' / f'fan{fan}_figure8_comparison.png'
        opt_cov = best_cov.squeeze(0).cpu().numpy()
        assert opt_cov.shape[-2:] == (3, 3), f'planned covariance must be 3x3, got {opt_cov.shape}'
        _plot(env, fan, sigma0, q_std, nominal_wps, nominal_curve, None,
              positions_xyz, opt_cov, plot_path, obstacles=FIG8_OBSTACLES)


def run_plan(fan: int, scenario: str = 'baseline', plot: bool = False) -> None:
    """Optimise and save waypoints for one fan level/scenario (optionally plot).

    Explicit 3-way dispatch to _run_plan_2d/_run_plan_3d/_run_plan_figure8 --
    everything scenario-specific (construction, optimizer state shape,
    waypoint conversion) lives in those functions and the planning_2d/
    planning_3d modules they call into; nothing here branches beyond this.
    """
    if scenario == 'gate':
        _run_plan_3d(fan, plot)
    elif scenario == 'figure8':
        _run_plan_figure8(fan, plot)
    else:
        _run_plan_2d(fan, plot)
