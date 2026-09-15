"""End-to-end smoke test of the differentiable pdSTL planning pipeline:

    v -> u -> (mu, Sigma) -> {N(mu_k, Sigma_k)} -> {[p_k, p_k]} -> Eventually -> J -> grad_v J

through one precise Gaussian belief model, with every shaping heuristic off.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from models.beliefs import GaussianBelief, create_gaussian_belief_trajectory
from pdstl.operators import LessThan
from planning.examples import (
    end_to_end_setup,
    load_end_to_end_config,
    run_end_to_end_reach,
)
from planning.planner import Planner
from utils import get_device
from visualization.robustness import plot_end_to_end


@pytest.fixture(scope="module")
def result():
    return run_end_to_end_reach(save=False, verbose=False)


def _assert_valid_covariance(cov):
    assert torch.isfinite(cov).all()
    torch.testing.assert_close(cov, cov.transpose(-1, -2), atol=1e-6, rtol=0)
    assert (torch.linalg.eigvalsh(cov) >= -1e-6).all()


def test_config_disables_every_shaping_heuristic():
    _, planner_cfg = load_end_to_end_config()
    assert planner_cfg["w_dist"] == planner_cfg["w_obs"] == planner_cfg["w_visit"] == 0
    assert planner_cfg["w_phi"] > 0
    assert planner_cfg["scale"] <= 0  # scored directly, not smoothed


def test_gradient_reaches_controls_through_the_whole_pipeline():
    cfg, planner_cfg = load_end_to_end_config()
    device = get_device()
    dyn, x0_mean, x0_cov, predicate, spec = end_to_end_setup(cfg, device)

    v = torch.zeros(cfg["H"], 2, device=device, requires_grad=True)
    mean, cov = dyn(v, x0_mean, x0_cov)
    traj = create_gaussian_belief_trajectory(mean[0], cov[0])
    assert all(isinstance(belief, GaussianBelief) for belief in traj)

    atomic = predicate(traj)
    robustness = spec(traj, scale=planner_cfg["scale"])[0, 0, 0]
    loss = Planner(dyn, None, cfg["H"], config=planner_cfg)._objective(
        mean, dyn.bound_control(v), robustness
    )
    loss.backward()

    assert torch.isfinite(robustness)
    assert torch.isfinite(mean).all()
    _assert_valid_covariance(cov[0])
    assert torch.equal(atomic[..., 0], atomic[..., 1])
    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    assert v.grad.abs().sum() > 0


def test_every_precise_belief_returns_equal_bounds():
    cfg, _ = load_end_to_end_config()
    device = get_device()
    dyn, x0_mean, x0_cov, predicate, _ = end_to_end_setup(cfg, device)

    mean, cov = dyn(torch.randn(cfg["H"], 2, device=device), x0_mean, x0_cov)
    traj = create_gaussian_belief_trajectory(mean[0], cov[0])

    for belief in traj:
        for event in (predicate, LessThan(cfg["threshold"], dim=cfg["dim"])):
            bounds = belief.probability_bounds(event)
            assert torch.equal(bounds[:, 0], bounds[:, 1])


def test_optimisation_improves_the_pdstl_score(result):
    assert result["score_final"] > result["score_initial"]
    # a meaningful improvement: from near-certain violation to likely satisfaction
    assert result["score_initial"] < 0.01
    assert result["score_final"] > 0.5
    assert all(torch.isfinite(torch.tensor(result["history"])))


def test_optimised_controls_respect_the_tanh_limits(result):
    controls = result["controls"]
    assert torch.isfinite(controls).all()
    assert (controls.abs() <= result["u_max"]).all()


def test_optimised_trajectory_moves_toward_the_predicate_region(result):
    cfg, _ = load_end_to_end_config()
    dim = cfg["dim"]
    before = result["mean_initial"][0, :, dim]
    after = result["mean_final"][0, :, dim]

    assert torch.isfinite(after).all()
    assert after.max() > before.max()
    assert (cfg["threshold"] - after).abs().min() < (cfg["threshold"] - before).abs().min()
    _assert_valid_covariance(result["cov_final"][0])


def test_atomic_bounds_coincide_before_and_after(result):
    for key in ("atomic_initial", "atomic_final"):
        atomic = result[key]
        assert torch.equal(atomic[..., 0], atomic[..., 1])
    assert result["interval_final"][0] == result["interval_final"][1]


def test_diagnostic_plot_draws_state_and_atomic_panels(result):
    cfg, _ = load_end_to_end_config()
    dim = cfg["dim"]
    time = [t * cfg["dt"] for t in range(cfg["H"] + 1)]

    def run(tag):
        return {
            "mean": result[f"mean_{tag}"][0, :, dim],
            "var": result[f"cov_{tag}"][0, :, dim, dim],
            "atomic": result[f"atomic_{tag}"],
            "score": result[f"score_{tag}"],
        }

    fig, axes = plot_end_to_end(time, run("initial"), run("final"), cfg["threshold"])

    assert len(axes) == 2
    fig.canvas.draw()
    plt.close(fig)
