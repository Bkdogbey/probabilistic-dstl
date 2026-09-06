import math

import pytest
import torch

from pdstl import Always, And, Eventually, Implies, Or, Predicate, Until


def bounds(values):
    return torch.tensor(values, dtype=torch.float64).reshape(1, -1, 2)


def test_predicate_preserves_tensor_and_gradient():
    trace = bounds([[0.2, 0.4], [0.5, 0.8]]).requires_grad_()
    formula = Predicate("a")
    assert formula({"a": trace}) is trace
    formula({"a": trace}).sum().backward()
    torch.testing.assert_close(trace.grad, torch.ones_like(trace))
    assert formula.robustness({"a": trace}).shape == (1, 1, 2)
    assert formula.robustness({"a": trace}, keepdim=False).shape == (1, 2)


@pytest.mark.parametrize("trace", [
    torch.ones(1, 2), torch.ones(1, 2, 3), torch.empty(1, 0, 2),
    bounds([[-0.1, 0.3]]), bounds([[0.8, 0.2]]), bounds([[0.5, 1.1]]),
    bounds([[math.nan, 0.8]]), bounds([[0.1, math.inf]]),
])
def test_invalid_bounds_rejected(trace):
    with pytest.raises(ValueError):
        Predicate("a")({"a": trace})


def test_input_contract_errors():
    formula = Predicate("a")
    with pytest.raises(TypeError):
        formula({"a": torch.ones(1, 1, 2, dtype=torch.int64)})
    with pytest.raises(ValueError):
        formula({})
    with pytest.raises(KeyError, match="missing"):
        formula({"b": bounds([[0, 1]])})
    with pytest.raises(ValueError, match="share"):
        formula({"a": bounds([[0, 1]]), "b": bounds([[0, 1], [0, 1]])})
    with pytest.raises(ValueError, match="share"):
        formula({"a": bounds([[0, 1]]), "b": bounds([[0, 1]]).float()})
    with pytest.raises(ValueError, match="time axis"):
        formula({"a": bounds([[0, 1]])}, keepdim=False)


@pytest.mark.parametrize("scale", [math.nan, math.inf, -math.inf, True, "2"])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        Predicate("a")({"a": bounds([[0.2, 0.4]])}, scale=scale)


def test_frechet_boolean_equations_without_independence():
    a, b = Predicate("a"), Predicate("b")
    inputs = {"a": bounds([[0.5, 0.5], [0.7, 0.9]]),
              "b": bounds([[0.5, 0.5], [0.6, 0.8]])}
    # Complementary events with marginals .5 can have intersection zero.
    torch.testing.assert_close(And(a, b)(inputs), bounds([[0, 0.5], [0.3, 0.8]]))
    torch.testing.assert_close(Or(a, b)(inputs), bounds([[0.5, 1], [0.7, 1]]))
    torch.testing.assert_close((~a)(inputs), bounds([[0.5, 0.5], [0.1, 0.3]]))
    torch.testing.assert_close(Implies(a, b)(inputs), ((~a) | b)(inputs))
    torch.testing.assert_close((~(a & b))(inputs), ((~a) | (~b))(inputs))


def test_until_uses_inclusive_prefix_and_frechet_combination():
    a, b = Predicate("a"), Predicate("b")
    inputs = {"a": bounds([[0.2, 0.2]]), "b": bounds([[0.9, 0.9]])}
    torch.testing.assert_close(Until(a, b, [0, 0])(inputs), bounds([[0.1, 0.2]]))


@pytest.mark.parametrize("interval", [None, [0, math.inf], [-1, 2], [2, 1],
                                      [0.5, 2], [True, 2], [1], "0,2"])
@pytest.mark.parametrize("kind", [Always, Eventually, Until])
def test_invalid_or_unbounded_intervals_rejected(interval, kind):
    a = Predicate("a")
    with pytest.raises(ValueError):
        if kind is Until:
            kind(a, a, interval)
        else:
            kind(a, interval)


def test_horizons_alignment_and_missing_future():
    a, b = Predicate("a"), Predicate("b")
    inputs = {"a": bounds([[0.4, 0.7]] * 8), "b": bounds([[0.6, 0.8]] * 8)}
    nested = Always(Eventually(a, [1, 2]), [0, 3])
    assert nested.horizon == 5
    assert nested(inputs).shape == (1, 3, 2)
    assert (nested & b)(inputs).shape == (1, 3, 2)
    until = Until(Always(a, [0, 1]), Eventually(b, [0, 2]), [1, 3])
    assert until.horizon == 5
    assert until(inputs).shape == (1, 3, 2)
    short = {name: trace[:, :5] for name, trace in inputs.items()}
    with pytest.raises(ValueError, match="at least 6"):
        nested(short)
    with pytest.raises(ValueError, match="at least 5"):
        Always(a, [3, 4])({"a": inputs["a"][:, :2]})


def test_interval_refinement_is_preserved():
    generator = torch.Generator().manual_seed(31)
    centers = {name: 0.25 + 0.5 * torch.rand(2, 8, generator=generator, dtype=torch.float64)
               for name in ("a", "b")}
    wide = {name: torch.stack([p - 0.2, p + 0.2], -1) for name, p in centers.items()}
    narrow = {name: torch.stack([p - 0.05, p + 0.05], -1) for name, p in centers.items()}
    a, b = Predicate("a"), Predicate("b")
    formulas = [~a, a & b, a | b, Implies(a, b), Always(a | b, [1, 3]),
                Eventually(a & b, [0, 2]), Until(~a, b, [1, 3])]
    for formula in formulas:
        outer, inner = formula(wide), formula(narrow)
        assert torch.all(outer[..., 0] <= inner[..., 0] + 1e-12)
        assert torch.all(inner[..., 1] <= outer[..., 1] + 1e-12)
        assert torch.all(outer[..., 0] >= 0)
        assert torch.all(outer[..., 1] <= 1)
        assert torch.all(outer[..., 0] <= outer[..., 1])
