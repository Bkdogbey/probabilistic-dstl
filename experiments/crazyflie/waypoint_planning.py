"""Offline pdSTL planning for the Crazyflie experiment.

Optimises a waypoint plan for one fan level and writes it to
waypoints/pdstl_fan<L>.json. Called by `run.py plan`; has no hardware/ROS
dependency (only torch/numpy, plus matplotlib for --plot).

Each fan level uses its own initial covariance Σ0 (config.SIGMA0_PER_FAN), so
both the optimisation and the saved plot reflect that fan's real uncertainty.

Planning is 3D: the optimizer jointly places (x, y, z) for every waypoint
(components/env3d.py), so there is no post-hoc altitude profile to apply here
-- waypoints come straight out of the optimized mean trajectory.
"""

from __future__ import annotations

import datetime

import numpy as np
import torch

from components.config import (
    DT,
    EXPERIMENT_DIR,
    FLIGHT_Z_BOUNDS,
    OBSTACLES,
    PLANNER_ALPHA,
    SAFE_PATH_FLIGHT_POINTS,
    SIGMA0_PER_FAN,
    T,
    Z_HEIGHT,
    BeliefTrajectory,
    Environment,
    FLIGHT_X_BOUNDS,
    FLIGHT_Y_BOUNDS,
    TorchGaussianBelief,
    build_planner,
    nominal_safe_waypoints,
    reference_direct_path,
    save_pdstl_waypoints,
    x0_belief,
)


def _validate_inside_flight_area(waypoints: list[tuple[float, float, float]]) -> None:
    x_min, x_max = FLIGHT_X_BOUNDS
    y_min, y_max = FLIGHT_Y_BOUNDS
    z_min, z_max = FLIGHT_Z_BOUNDS
    outside = [
        (idx, x, y, z)
        for idx, (x, y, z) in enumerate(waypoints)
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max)
    ]
    if outside:
        details = ', '.join(f'#{idx}=({x:.3f}, {y:.3f}, {z:.3f})' for idx, x, y, z in outside)
        raise ValueError(
            'Generated waypoint(s) outside flight area '
            f'x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]: {details}'
        )


def _nominal_init_guess() -> torch.Tensor:
    """Convert the deterministic safe path to T velocity controls for warm-starting.

    The nominal path flies at constant Z_HEIGHT, so its z-velocity column is
    all zeros -- the optimizer is free to move away from that in z.
    """
    wps = np.array(nominal_safe_waypoints(n_points=T))
    vels_xy = np.diff(wps[:, :2], axis=0) / DT
    vels_xy = np.vstack([vels_xy, [[0.0, 0.0]]])
    vels_z = np.zeros((vels_xy.shape[0], 1))
    vels = np.hstack([vels_xy, vels_z])
    init_u = torch.tensor(vels, dtype=torch.float32)
    assert init_u.shape[-1] == 3, f'init_u must be [T, 3], got {tuple(init_u.shape)}'
    return init_u


def _evaluate_rho(planner, init_u: torch.Tensor,
                  x0_mean: torch.Tensor, x0_cov: torch.Tensor) -> float:
    """Evaluate STL P(sat) for a given control sequence without optimising."""
    u_norm = torch.clamp(init_u / planner.dyn.u_max, -0.99, 0.99)
    v = 0.5 * torch.log((1 + u_norm) / (1 - u_norm))
    with torch.no_grad():
        mean_trace, cov_trace = planner.dyn(v, x0_mean, x0_cov)
        beliefs = [TorchGaussianBelief(mean_trace[:, t, :], cov_trace[:, t]) for t in range(T + 1)]
        phi = planner.env.get_specification(T)
        return phi(BeliefTrajectory(beliefs))[0, 0, 0].item()


# ── Plotting ─────────────────────────────────────────────────────────────────
def _obstacle_clearance(curve_xyz: np.ndarray, obs: dict) -> float:
    """Min distance from any point on curve_xyz [N,3] to an {'x','y','z'} obstacle box (0 = inside).

    Correctly reads as positive when flown over the top of a box (large z,
    dz > 0) even with zero lateral (x,y) clearance, not just when routed
    around it.
    """
    x0, x1 = obs['x']
    y0, y1 = obs['y']
    z0, z1 = obs.get('z', (curve_xyz[:, 2].min(), curve_xyz[:, 2].max()))
    dx = np.maximum(np.maximum(x0 - curve_xyz[:, 0], 0.0), curve_xyz[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - curve_xyz[:, 1], 0.0), curve_xyz[:, 1] - y1)
    dz = np.maximum(np.maximum(z0 - curve_xyz[:, 2], 0.0), curve_xyz[:, 2] - z1)
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


def _draw_env(ax, env: Environment) -> None:
    """Draw the arena (workspace bounds wireframe, obstacle boxes, goal box) in 3D."""
    ax.set_xlim(FLIGHT_X_BOUNDS[0] - 0.1, FLIGHT_X_BOUNDS[1] + 0.1)
    ax.set_ylim(FLIGHT_Y_BOUNDS[0] - 0.1, FLIGHT_Y_BOUNDS[1] + 0.1)
    ax.set_zlim(FLIGHT_Z_BOUNDS[0] - 0.1, FLIGHT_Z_BOUNDS[1] + 0.1)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    _draw_box(
        ax, FLIGHT_X_BOUNDS, FLIGHT_Y_BOUNDS, FLIGHT_Z_BOUNDS,
        facecolor='none', edgecolor='black', alpha=0.15, linestyle='dashed',
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


def _plot(env, fan, sigma0, nominal_wps, nominal_curve, reference_curve,
          opt_xyz, opt_cov, out_path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 7.5))
    ax_before = fig.add_subplot(1, 2, 1, projection='3d')
    ax_after = fig.add_subplot(1, 2, 2, projection='3d')
    fig.suptitle(f'Fan {fan}  (Σ0 = {sigma0} m²)', fontsize=13)

    # ── Before: deterministic safe path, obstacle-free reference + clearances ──
    _draw_env(ax_before, env)
    ax_before.set_title('Before (Deterministic Safe Path)')
    ax_before.plot(reference_curve[:, 0], reference_curve[:, 1], reference_curve[:, 2],
                   'k--', lw=1.2, alpha=0.6, label='reference (no obstacles)')
    ax_before.plot(nominal_curve[:, 0], nominal_curve[:, 1], nominal_curve[:, 2],
                   'b-', lw=1.5, alpha=0.8)
    ax_before.scatter(nominal_wps[:, 0], nominal_wps[:, 1], nominal_wps[:, 2],
                      c='blue', s=25, label='flown waypoints')
    ax_before.scatter(*nominal_wps[0], c='green', s=80, label='start')
    ax_before.scatter(*nominal_wps[-1], c='red', s=80, marker='s', label='end')
    for obs in OBSTACLES:
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
    ax_after.plot(reference_curve[:, 0], reference_curve[:, 1], reference_curve[:, 2],
                  'k--', lw=1.2, alpha=0.6)
    ax_after.plot(opt_xyz[:, 0], opt_xyz[:, 1], opt_xyz[:, 2], 'b.-', lw=2, ms=8)
    ax_after.scatter(*opt_xyz[0], c='green', s=80, label='start')
    ax_after.scatter(*opt_xyz[-1], c='red', s=80, marker='s', label='end')
    for obs in OBSTACLES:
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
def run_plan(fan: int, plot: bool = False) -> None:
    """Optimise and save waypoints for one fan level (optionally plot)."""
    sigma0 = SIGMA0_PER_FAN[fan]
    print(f'Planning for fan {fan}  (Σ0 = {sigma0} m², T = {T})')

    planner, _dynamics, env = build_planner(fan)
    x0_mean, x0_cov = x0_belief(fan)
    init_u = _nominal_init_guess()

    rho_before = _evaluate_rho(planner, init_u, x0_mean, x0_cov)
    print(f'rho_before (deterministic safe path): {rho_before:.4f}')

    best_mean, best_cov, _best_u, best_p, _history = planner._optimize_window(
        x0_mean, x0_cov, init_guess=init_u, verbose=True,
    )
    print(f'rho_after  (optimised):               {best_p:.4f}')

    positions_xyz = best_mean.squeeze(0).cpu().numpy()
    assert positions_xyz.shape[-1] == 3, f'planned positions must be 3D, got {positions_xyz.shape}'
    waypoints = [(float(x), float(y), float(z)) for x, y, z in positions_xyz]
    _validate_inside_flight_area(waypoints)

    meta = {
        'sigma0': sigma0,
        'rho_before': round(rho_before, 4),
        'rho_after': round(float(best_p), 4),
        'alpha': PLANNER_ALPHA,
        'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }
    out = save_pdstl_waypoints(fan, waypoints, meta)
    print(f'Wrote {len(waypoints)} waypoints to {out}')

    if plot:
        nominal_wps = np.array(nominal_safe_waypoints(n_points=SAFE_PATH_FLIGHT_POINTS))
        nominal_curve = np.array(nominal_safe_waypoints(n_points=200))
        reference_xy = np.array(reference_direct_path(n_points=200))
        reference_curve = np.column_stack([reference_xy, np.full(len(reference_xy), Z_HEIGHT)])
        plot_path = EXPERIMENT_DIR / 'plots' / f'fan{fan}_comparison.png'
        opt_cov = best_cov.squeeze(0).cpu().numpy()
        assert opt_cov.shape[-2:] == (3, 3), f'planned covariance must be 3x3, got {opt_cov.shape}'
        _plot(env, fan, sigma0, nominal_wps, nominal_curve, reference_curve,
              positions_xyz, opt_cov, plot_path)
