"""Offline pdSTL planning for the Crazyflie experiment.

Optimises a waypoint plan for one fan level and writes it to
waypoints/pdstl_fan<L>.json. Called by `run.py plan`; has no hardware/ROS
dependency (only torch/numpy, plus matplotlib for --plot).

Each fan level uses its own initial covariance Σ0 (config.SIGMA0_PER_FAN), so
both the optimisation and the saved plot reflect that fan's real uncertainty.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import torch

from components.config import (
    DT,
    EXPERIMENT_DIR,
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


def z_profile(xy_waypoints: np.ndarray) -> list[float]:
    """Altitude for each waypoint. Constant for now (2D planning at fixed height).

    For 3D flights later, replace this with a real profile — nothing else needs
    restructuring.
    """
    return [Z_HEIGHT] * len(xy_waypoints)


def _validate_inside_flight_area(waypoints: list[tuple[float, float, float]]) -> None:
    x_min, x_max = FLIGHT_X_BOUNDS
    y_min, y_max = FLIGHT_Y_BOUNDS
    outside = [
        (idx, x, y)
        for idx, (x, y, _z) in enumerate(waypoints)
        if not (x_min <= x <= x_max and y_min <= y <= y_max)
    ]
    if outside:
        details = ', '.join(f'#{idx}=({x:.3f}, {y:.3f})' for idx, x, y in outside)
        raise ValueError(
            'Generated waypoint(s) outside flight area '
            f'x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]: {details}'
        )


def _nominal_init_guess() -> torch.Tensor:
    """Convert the deterministic safe path to T velocity controls for warm-starting."""
    wps = np.array(nominal_safe_waypoints(n_points=T))
    vels = np.diff(wps[:, :2], axis=0) / DT
    vels = np.vstack([vels, [[0.0, 0.0]]])
    return torch.tensor(vels, dtype=torch.float32)


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
def _obstacle_clearance(curve_xy: np.ndarray, obs: dict) -> float:
    """Min distance from any point on curve_xy [N,2] to an {'x','y'} obstacle box (0 = inside)."""
    x0, x1 = obs['x']
    y0, y1 = obs['y']
    dx = np.maximum(np.maximum(x0 - curve_xy[:, 0], 0.0), curve_xy[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - curve_xy[:, 1], 0.0), curve_xy[:, 1] - y1)
    return float(np.min(np.sqrt(dx**2 + dy**2)))


def _draw_env(ax, env: Environment) -> None:
    import matplotlib.patches as patches

    ax.set_aspect('equal')
    ax.set_xlim(FLIGHT_X_BOUNDS[0] - 0.1, FLIGHT_X_BOUNDS[1] + 0.1)
    ax.set_ylim(FLIGHT_Y_BOUNDS[0] - 0.1, FLIGHT_Y_BOUNDS[1] + 0.1)
    ax.grid(True, alpha=0.3)

    ax.add_patch(patches.Rectangle(
        (FLIGHT_X_BOUNDS[0], FLIGHT_Y_BOUNDS[0]),
        FLIGHT_X_BOUNDS[1] - FLIGHT_X_BOUNDS[0],
        FLIGHT_Y_BOUNDS[1] - FLIGHT_Y_BOUNDS[0],
        facecolor='none', edgecolor='black', linestyle='--', linewidth=1.2,
        label='flight area',
    ))
    for obs in env.obstacles:
        ox, oy = obs['x'], obs['y']
        ax.add_patch(patches.Rectangle(
            (ox[0], oy[0]), ox[1] - ox[0], oy[1] - oy[0],
            facecolor='red', alpha=0.4, edgecolor='darkred',
        ))
    if env.goal:
        gx, gy = env.goal['x'], env.goal['y']
        ax.add_patch(patches.Rectangle(
            (gx[0], gy[0]), gx[1] - gx[0], gy[1] - gy[0],
            facecolor='green', alpha=0.3, edgecolor='darkgreen',
        ))


def _plot(env, fan, sigma0, nominal_wps, nominal_curve, reference_curve,
          opt_xy, opt_cov, out_path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'Fan {fan}  (Σ0 = {sigma0} m²)', fontsize=13)

    # ── Before: deterministic safe path, obstacle-free reference + clearances ──
    _draw_env(ax_before, env)
    ax_before.set_title('Before (Deterministic Safe Path)')
    ax_before.plot(reference_curve[:, 0], reference_curve[:, 1],
                   'k--', lw=1.2, alpha=0.6, label='reference (no obstacles)')
    ax_before.plot(nominal_curve[:, 0], nominal_curve[:, 1], 'b-', lw=1.5, alpha=0.8)
    ax_before.plot(nominal_wps[:, 0], nominal_wps[:, 1], 'bo', ms=5, label='flown waypoints')
    ax_before.plot(nominal_wps[0, 0], nominal_wps[0, 1], 'go', ms=10, label='start')
    ax_before.plot(nominal_wps[-1, 0], nominal_wps[-1, 1], 'rs', ms=10, label='end')
    for obs in OBSTACLES:
        clearance = _obstacle_clearance(nominal_curve, obs)
        cx = (obs['x'][0] + obs['x'][1]) / 2
        cy = (obs['y'][0] + obs['y'][1]) / 2
        ax_before.annotate(f"{obs['name']}\n{clearance * 100:.1f} cm",
                           (cx, cy), ha='center', va='center', fontsize=8, color='darkred')
    ax_before.legend(fontsize=8)

    # ── After: pdSTL-optimised plan, with this fan's belief-covariance ellipses ──
    _draw_env(ax_after, env)
    ax_after.set_title('After (pdSTL Optimised)')
    ax_after.plot(reference_curve[:, 0], reference_curve[:, 1], 'k--', lw=1.2, alpha=0.6)
    ax_after.plot(opt_xy[:, 0], opt_xy[:, 1], 'b.-', lw=2, ms=8)
    ax_after.plot(opt_xy[0, 0], opt_xy[0, 1], 'go', ms=10, label='start')
    ax_after.plot(opt_xy[-1, 0], opt_xy[-1, 1], 'rs', ms=10, label='end')
    for t in range(len(opt_xy)):
        vals, vecs = np.linalg.eigh(opt_cov[t])
        angle = math.degrees(math.atan2(*vecs[:, -1][::-1]))
        w, h = 2 * 2 * np.sqrt(np.maximum(vals, 0))  # 2σ ellipse (diameter)
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

    positions_xy = best_mean.squeeze(0).cpu().numpy()
    z_values = z_profile(positions_xy)
    waypoints = [(float(x), float(y), z) for (x, y), z in zip(positions_xy, z_values)]
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
        nominal_wps = np.array(nominal_safe_waypoints(n_points=SAFE_PATH_FLIGHT_POINTS))[:, :2]
        nominal_curve = np.array(nominal_safe_waypoints(n_points=200))[:, :2]
        reference_curve = np.array(reference_direct_path(n_points=200))
        plot_path = EXPERIMENT_DIR / 'plots' / f'fan{fan}_comparison.png'
        _plot(env, fan, sigma0, nominal_wps, nominal_curve, reference_curve,
              positions_xy, best_cov.squeeze(0).cpu().numpy(), plot_path)