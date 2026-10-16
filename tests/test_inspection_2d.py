"""2D inspection: map, specs R1-R3 and their parts on noise-free paths."""

import pytest
import torch

from experiments import inspection_2d as exp
from utils import load_config

START, SLOT, P1 = (1.0, 1.0), (4.0, 1.5), (6.0, 1.5)
C, D = (8.1, 4.6), (8.6, 8.6)


@pytest.fixture(scope="module")
def specs():
    cfg = load_config(exp.CONFIG)
    return exp.build_specs(exp.build_environment(cfg), cfg), cfg["H"]


def _path(legs, H):
    """Hold each point for its count of samples; stay at the last one."""
    points = [point for point, count in legs for _ in range(count)]
    return points + [points[-1]] * (H + 1 - len(points))


def _truth(specs, legs):
    specs, H = specs
    path = _path(legs, H)
    return {
        name: (
            bool(exp.satisfied(formula, path)),
            {k: bool(exp.satisfied(p, path)) for k, p in parts.items()},
        )
        for name, (formula, parts) in specs.items()
    }


GOOD = [(START, 1), (SLOT, 1), (P1, 11), (C, 1), (D, 1)]
CASES = {
    "good": (GOOD, (True, True, True)),
    "C before inspection": (
        [(START, 1), (SLOT, 1), (C, 1), *GOOD[2:]],
        (True, True, False),
    ),
    "nine dwell steps": (
        [(START, 1), (SLOT, 1), (P1, 10), (C, 1), (D, 1)],
        (True, False, False),
    ),
    "D before C": (
        [(START, 1), (SLOT, 1), (P1, 11), (D, 1), (C, 1), (D, 1)],
        (True, True, False),
    ),
    "never leaves": ([(START, 1)], (False, False, False)),
}


@pytest.mark.parametrize("case", CASES)
def test_specs_and_parts_agree(specs, case):
    legs, expected = CASES[case]
    truth = _truth(specs, legs)
    assert tuple(truth[name][0] for name in ("R1", "R2", "R3")) == expected
    for holds, parts in truth.values():
        assert holds == all(parts.values())


def test_staying_home_is_an_inspection_failure_not_ordering(specs):
    parts = _truth(specs, [(START, 1)])["R3"][1]
    assert parts == {
        "safety": True,
        "ordering": True,
        "inspection": False,
        "deadline": False,
    }


def test_unsafe_start_raises():
    cfg = load_config(exp.CONFIG)
    cfg["x0_mean"] = [4.0, 3.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="not safe"):
        exp.build_environment(cfg)


def test_slot_route_seed_holds_in_p1_and_satisfies_r3():
    cfg = load_config(exp.CONFIG)
    env = exp.build_environment(cfg)
    r3 = exp.build_specs(env, cfg)["R3"][0]
    length, _, _ = exp.prescribed_route(env, cfg, ["P1"])
    slot, _, _ = exp.prescribed_route(env, cfg, cfg["routes"]["slot"]["stops"])
    assert 5.0 < length < slot
    s = exp.problem(cfg, **cfg["nominal"])
    reference, controls = exp.route_seed(env, cfg, s.dyn, "slot")
    assert reference.shape == (cfg["H"] + 1, 2)
    assert controls.abs().max() <= s.dyn.u_max
    s.spec = r3
    mean = exp._mean(s, controls).rollout.aux["mean_trace"][0]
    assert exp.longest_stay(env, mean, "P1") > cfg["dwell"]
    assert exp.satisfied(r3, mean)


def test_failure_class_is_the_first_failing_part():
    flags = {
        "safety": torch.tensor([True, True, False, True]),
        "ordering": torch.tensor([True, False, False, True]),
        "deadline": torch.tensor([True, False, True, True]),
    }
    classes = exp.failure_classes(flags)
    assert classes.tolist() == ["", "ordering", "safety", ""]


def test_slot_traversal_is_read_from_the_mean_path():
    cfg = load_config(exp.CONFIG)
    env = exp.build_environment(cfg)
    assert exp.slot_traversed(env, _path(GOOD, cfg["H"]))
    assert not exp.slot_traversed(env, _path([(START, 1), (C, 1)], 3))
