"""Pickup-delivery study: independent checker, semantics references, diagnostics."""

import numpy as np
import pytest
import torch

from baselines.det_stl import from_pdstl
from experiments import pickup_delivery as pdl
from experiments import pickup_delivery_study as study
from experiments import task_checker as tc
from models.rollouts import (
    create_belief_trajectory,
    create_gaussian_belief_trajectory,
    gaussian_rollout,
    sample_trajectories,
)
from pdstl.predicates import AxisInterval
from planning.environment import inside, outside
from utils import load_config

# --- Independent checker ---------------------------------------------------------

H, D = 20, 3  # small horizon and dwell: a dwell spans D + 1 samples
TASK = {
    "workspace": (0.0, 10.0, 0.0, 10.0),
    "obstacles": [(4.0, 6.0, 0.0, 4.0)],
    "pickups": {"P1": (1.0, 2.0, 6.0, 7.0), "P2": (7.0, 8.0, 1.0, 2.0)},
    "scanner": (8.0, 9.0, 6.0, 7.0),
    "delivery": (8.0, 9.0, 8.5, 9.5),
}
AT = {
    "free": (0.5, 0.5),
    "P1": (1.5, 6.5),
    "P2": (7.5, 1.5),
    "C": (8.5, 6.5),
    "D": (8.5, 9.0),
    "rack": (5.0, 2.0),
    "out": (-1.0, 5.0),
}


def trace(*segments):
    """[1, H + 1, 2] positions from (place, samples) segments, padded free."""
    points = [AT[place] for place, n in segments for _ in range(n)]
    points += [AT["free"]] * (H + 1 - len(points))
    assert len(points) == H + 1
    return np.array([points])


def verdict(xy, spec="R3"):
    out = tc.check(xy, TASK, H, D, spec)
    return {key: value[0] for key, value in out.items()}


@pytest.mark.parametrize("pickup", ["P1", "P2"])
def test_valid_load_scan_delivery_on_either_branch(pickup):
    out = verdict(trace(("free", 2), (pickup, 4), ("C", 2), ("D", 2)))
    assert out["success"] and out["failures"] == 0
    assert out["pickup"] == pickup
    assert (out["load_time"], out["scan_time"], out["delivery_time"]) == (
        2,
        6,
        8,
    )


@pytest.mark.parametrize(
    "segments, failures",
    [
        ([("C", 1), ("P1", 4), ("C", 1), ("D", 1)], ["SCAN_BEFORE_LOAD"]),
        ([("P1", 4), ("D", 1), ("C", 1), ("D", 1)], ["DELIVER_BEFORE_SCAN"]),
        # Seven samples in P1, never four in a row.
        (
            [("P1", 3), ("free", 1), ("P1", 3), ("C", 1), ("D", 1)],
            ["NO_LOAD"],
        ),
        ([("P1", 4), ("rack", 1), ("C", 1), ("D", 1)], ["OBSTACLE"]),
        ([("P1", 4), ("out", 1), ("C", 1), ("D", 1)], ["WORKSPACE"]),
        ([("P1", 4), ("C", 1)], ["NO_DELIVERY"]),
        ([("P1", 4), ("D", 1)], ["NO_SCAN"]),
        ([("rack", 1), ("P1", 4), ("C", 1)], ["OBSTACLE", "NO_DELIVERY"]),
    ],
)
def test_each_violation_sets_its_own_bit(segments, failures):
    out = verdict(trace(*segments))
    assert not out["success"]
    assert tc.failure_names(out["failures"]) == failures
    assert out["primary"] == failures[0]


def test_latest_admissible_dwell_and_the_horizon_boundary():
    latest = trace(("D", 1), ("free", H - D - 1), ("P2", D + 1))
    out = verdict(latest, "R2")
    assert out["success"] and out["load_time"] == H - D
    late = trace(("D", 1), ("free", H - D), ("P2", D))
    assert tc.failure_names(verdict(late, "R2")["failures"]) == ["NO_LOAD"]


def test_specs_mask_the_requirements_they_do_not_have():
    reach_only = trace(("free", 2), ("D", 1))
    assert verdict(reach_only, "R1")["success"]
    assert verdict(reach_only, "R2")["primary"] == "NO_LOAD"
    unordered = trace(("C", 1), ("P1", 4), ("D", 1))
    assert verdict(unordered, "R2")["success"]
    assert verdict(unordered, "R3")["primary"] == "SCAN_BEFORE_LOAD"


# --- Hard pdSTL against an independent reference of the defined semantics ----


def _conj(a, b):
    return np.stack(
        [
            np.maximum(0, a[..., 0] + b[..., 0] - 1),
            np.minimum(a[..., 1], b[..., 1]),
        ],
        -1,
    )


def _disj(a, b):
    return np.stack(
        [
            np.maximum(a[..., 0], b[..., 0]),
            np.minimum(1, a[..., 1] + b[..., 1]),
        ],
        -1,
    )


def _neg(a):
    return np.stack([1 - a[..., 1], 1 - a[..., 0]], -1)


def _window(a, lo, hi, reduce):
    n = len(a) - hi
    return np.array([reduce(a[t + lo : t + hi + 1], axis=0) for t in range(n)])


def _until(phi, psi, lo, hi):
    n = min(len(phi), len(psi))
    out = []
    for t in range(n - hi):
        candidates = [
            _conj(phi[t : tau + 1].min(axis=0), psi[tau])
            for tau in range(t + lo, t + hi + 1)
        ]
        out.append(np.max(candidates, axis=0))
    return np.array(out)


def reference(atoms, regions, horizon, dwell, spec_id):
    """Fréchet Boolean, min/max windows, strong inclusive Until (numpy)."""

    def inside(name):
        x, y = atoms[name]
        return _conj(x, y)

    safe = inside("bounds")
    for obstacle in ("rack_low", "rack_high"):
        safe = _conj(safe, _neg(inside(obstacle)))
    safe = _window(safe, 0, horizon, np.min)
    deliver = _window(inside("D"), 0, horizon, np.max)
    load = _disj(
        _window(inside("P1"), 0, dwell, np.min),
        _window(inside("P2"), 0, dwell, np.min),
    )
    if spec_id == "R1":
        return _conj(safe[0], deliver[0])
    if spec_id == "R2":
        loaded = _window(load, 0, horizon - dwell, np.max)
        return _conj(_conj(safe[0], loaded[0]), deliver[0])
    scanner = inside("C")
    loaded = _until(_neg(scanner), load, 0, horizon - dwell)
    scanned = _until(_neg(inside("D")), scanner, 0, horizon)
    return _conj(_conj(_conj(safe[0], loaded[0]), scanned[0]), deliver[0])


@pytest.fixture(scope="module")
def cfg():
    return load_config(pdl_config())


def pdl_config():
    return "configs/scenarios/pickup_delivery/warehouse.yaml"


@pytest.fixture(scope="module")
def nominal(cfg):
    regions = {k: pdl._box(v) for k, v in cfg["regions"].items()}
    return {"id": -1, "start": cfg["x0_mean"][:2], "regions": regions}


def random_atoms(env, steps, generator):
    """Random [steps, 2] probability intervals for every region's axis atoms.

    Regions that share an axis interval share that event (racks share x).
    """
    atoms, traces = {}, {}
    for name, region in env.regions.items():
        pair = []
        for dim, (low, high) in enumerate((region.x, region.y)):
            event = AxisInterval(low, high, dim=dim).name
            if event not in traces:
                bounds = np.sort(generator.uniform(0.5, 1, (steps, 2)), 1)
                if generator.uniform() < 0.3:
                    bounds = 1 - bounds[:, ::-1]  # some atoms mostly false
                traces[event] = bounds
            pair.append(traces[event])
        atoms[name] = pair
    return atoms, create_belief_trajectory(traces, dtype=torch.float64)


@pytest.mark.parametrize("spec_id", pdl.SPECS)
def test_hard_scores_match_the_independent_reference(cfg, nominal, spec_id):
    env = pdl.build_environment(cfg, nominal)
    horizon, dwell = 8, 2
    spec = pdl.specification(env, horizon, dwell, spec_id)
    generator = np.random.default_rng(3)
    for _ in range(20):
        atoms, beliefs = random_atoms(env, horizon + 1, generator)
        got = spec(beliefs)[0, 0].numpy()
        want = reference(atoms, env.regions, horizon, dwell, spec_id)
        np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("spec_id", pdl.SPECS)
def test_lookahead_fits_exactly_inside_the_horizon(cfg, nominal, spec_id):
    env = pdl.build_environment(cfg, nominal)
    spec = pdl.specification(env, 8, 2, spec_id)
    assert pdl.check_lookahead(spec, 8) == 8
    with pytest.raises(ValueError):
        pdl.check_lookahead(spec, 7)
    atoms, beliefs = random_atoms(env, 9, np.random.default_rng(0))
    assert spec(beliefs).shape[1] == 1  # one complete origin, no sample past H


@pytest.mark.parametrize("beta", [5.0, 50.0, 500.0])
def test_smooth_lower_never_exceeds_the_hard_lower(cfg, nominal, beta):
    env = pdl.build_environment(cfg, nominal)
    generator = np.random.default_rng(1)
    for spec_id in pdl.SPECS:
        spec = pdl.specification(env, 8, 2, spec_id)
        for _ in range(10):
            _, beliefs = random_atoms(env, 9, generator)
            hard = spec.probability_interval(beliefs)
            smooth = spec(beliefs, beta=beta)[0, 0]
            assert smooth[0] <= hard[0] + 1e-12
            assert smooth[1] >= hard[1] - 1e-12


# --- Gradients and execution model ----------------------------------------------


@pytest.fixture(scope="module")
def scene(cfg):
    instance, _ = pdl.draw_instance(cfg, 0)
    return pdl.build(cfg, instance, "medium")


@pytest.mark.parametrize("spec_id", ["R1", "R3"])
def test_directional_finite_differences_match_autograd(cfg, spec_id):
    """Smooth objective in float64 at a start-bank point (no active clamps)."""
    s = pdl.build(cfg, pdl.draw_instance(cfg, 0)[0], "medium")
    start = pdl.start_bank(s)["P1-slot"].controls.double()
    dyn = s.dyn.double()
    rollout = gaussian_rollout(dyn, *(x.double() for x in s.state))
    spec = pdl.specification(s.env, s.cfg["H"], cfg["dwell"], spec_id)
    point = torch.atanh(start / dyn.u_max * 0.95)
    generator = torch.Generator().manual_seed(0)
    direction = torch.randn(
        point.shape, dtype=torch.float64, generator=generator
    )
    direction /= direction.norm()

    def value(v):
        return spec.smooth_lower(rollout(v).belief_trajectory, 50.0)

    v = point.clone().requires_grad_(True)
    value(v).backward()
    analytic = float(v.grad.flatten() @ direction.flatten())
    eps = 1e-5
    with torch.no_grad():
        numeric = float(
            (value(point + eps * direction) - value(point - eps * direction))
            / (2 * eps)
        )
    scale = max(abs(analytic), 1e-8)
    assert abs(analytic - numeric) / scale < 1e-4, (analytic, numeric)


def test_sampled_execution_matches_the_predicted_moments(scene):
    controls = pdl.start_bank(scene)["P1-slot"].controls
    paths = sample_trajectories(
        scene.dyn,
        *scene.state,
        controls,
        20000,
        torch.Generator().manual_seed(0),
    )
    # Open-loop covariance does not depend on the controls.
    zeros = torch.zeros(scene.cfg["H"], 2)
    cov = scene.rollout(zeros).aux["cov_trace"][0, -1]
    mean = torch.as_tensor(pdl.mean_positions(scene, controls)[0, -1])
    final = paths[:, -1]
    std = cov.diagonal()[:2].sqrt()
    error = (final[:, :2].mean(0) - mean).abs()
    assert (error < 5 * std / 20000**0.5).all()
    empirical = torch.cov(final.T)
    sampling = 5 * float(cov.diagonal().max()) * (2 / 20000) ** 0.5
    torch.testing.assert_close(empirical, cov, rtol=0.0, atol=sampling)


# --- Margin baseline ---------------------------------------------------------------


def test_margin_inflates_obstacles_and_shrinks_admissible_regions(scene):
    spec = pdl.specification(
        scene.env, scene.cfg["H"], scene.cfg["dwell"], "R1"
    )
    beliefs = scene.rollout(torch.zeros(scene.cfg["H"], 2)).belief_trajectory
    nominal = from_pdstl(spec)
    torch.testing.assert_close(
        from_pdstl(spec, margin_k=0.0)(beliefs), nominal(beliefs)
    )
    sigma = beliefs.trace.covariance[0, :, 0, 0].sqrt()
    point = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    cov = torch.diag(torch.tensor([0.04, 0.04, 0.0, 0.0]))[None, None]
    single = create_gaussian_belief_trajectory(point[0], cov[0])
    rack = scene.env.region("rack_low")
    for atom in (outside(rack), inside(scene.env.region("D"))):
        rho0 = from_pdstl(atom)(single)
        rho2 = from_pdstl(atom, margin_k=2.0)(single)
        torch.testing.assert_close(rho2, rho0 - 2 * 0.2)
    assert sigma[-1] > sigma[0]  # the margin grows along the horizon


# --- Records -----------------------------------------------------------------------


def test_resume_refuses_a_changed_config(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(study, "OUTPUTS", tmp_path)
    block = cfg["blocks"]["smoke"]
    directory, manifest = study.open_experiment("x", cfg, "smoke", block)
    study.write_manifest(directory, manifest)
    assert study.open_experiment("x", cfg, "smoke", block)[0] == directory
    changed = {**cfg, "noise": {**cfg["noise"], "low": 0.0011}}
    with pytest.raises(SystemExit, match="config hash"):
        study.open_experiment("x", changed, "smoke", block)


def test_trial_keys_and_execution_seeds_are_shared_across_methods(cfg):
    tasks = study.trial_tasks(cfg, cfg["blocks"]["pilot"])
    keys = [task["key"] for task in tasks]
    assert len(keys) == len(set(keys))
    seeds = {}
    for task in tasks:
        condition = (task["instance"], task["noise"], task["spec"])
        seeds.setdefault(condition, set()).add(task["eval_seed"])
    assert all(len(values) == 1 for values in seeds.values())
