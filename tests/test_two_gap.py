"""Two-gap experiment: config, gaps, selection rule and Monte Carlo seed."""

import pytest
import torch

from experiments import two_gap
from experiments.reach_avoid import straight_line_guess
from utils import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config(two_gap.CONFIG)


def test_config_loads_with_three_doubling_noise_levels(cfg):
    assert cfg["dynamics"] == "double_integrator"
    assert list(cfg["noise"]) == ["low", "medium", "high"]
    levels = list(cfg["noise"].values())
    assert levels == sorted(levels)
    assert set(cfg["gaps"]) == {"slit", "around"}


@pytest.mark.parametrize("gap", ["slit", "around"])
def test_each_initial_guess_passes_through_its_gap(cfg, gap):
    s = two_gap.problem(cfg, "stlcg", None)
    controls = straight_line_guess(s.cfg, s.dyn, cfg["gaps"][gap])
    state, means = s.state[0], [s.state[0]]
    for control in controls:
        state = s.dyn.A @ state + s.dyn.B @ control
        means.append(state)
    assert two_gap.gap_used(cfg, torch.stack(means).numpy()) == gap


def test_stlcg_plans_the_mean_without_noise_and_pdstl_with_it(cfg):
    stlcg = two_gap.problem(cfg, "stlcg", "high")
    assert torch.count_nonzero(stlcg.dyn.Q) == 0
    assert torch.count_nonzero(stlcg.state[1]) == 0
    pdstl = two_gap.problem(cfg, "pdstl", "high")
    q = cfg["noise"]["high"]
    torch.testing.assert_close(pdstl.dyn.Q, q**2 * torch.eye(4))


def _row(gap, robustness, score, effort):
    return {
        "gap": gap,
        "robustness": robustness,
        "score": score,
        "effort": effort,
    }


def test_selection_meets_the_requirement_then_least_effort(cfg):
    slit = _row("slit", 0.30, 0.90, 10.0)
    around = _row("around", 0.45, 0.97, 20.0)
    # STLCG: both clear 5 cm on the mean, so the cheaper slit.
    assert two_gap.select(cfg, "stlcg", [slit, around]) is slit
    # pdSTL: only the way around reaches 0.95.
    assert two_gap.select(cfg, "pdstl", [slit, around]) is around


def test_selection_keeps_the_best_criterion_when_none_qualifies(cfg):
    a = _row("slit", 0.01, 0.80, 10.0)
    b = _row("around", 0.03, 0.90, 20.0)
    assert two_gap.select(cfg, "stlcg", [a, b]) is b
    assert two_gap.select(cfg, "pdstl", [a, b]) is b


def test_clearance_is_distance_to_the_nearest_block():
    blocks = [{"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}]
    paths = torch.tensor([[[2.0, 0.5, 0, 0], [1.5, 0.5, 0, 0]]])
    torch.testing.assert_close(
        two_gap.clearance(paths, blocks), torch.tensor([0.5])
    )


def test_monte_carlo_draws_are_fixed_by_the_config_seed(cfg):
    assert cfg["monte_carlo"]["seed"] == 7
    s = two_gap.problem(cfg, "stlcg", None)
    controls = straight_line_guess(s.cfg, s.dyn, cfg["gaps"]["slit"])

    def run(seed):
        small = {**cfg, "monte_carlo": {"samples": 200, "seed": seed}}
        return two_gap.evaluate(small, controls, "high")

    assert run(7) == run(7)  # every plan sees the same draws
    assert run(7)["clearance_median"] != run(8)["clearance_median"]


def test_effort_is_the_sum_of_squared_controls():
    assert two_gap.effort(torch.tensor([[1.0, 2.0], [0.0, 3.0]])) == 14.0
