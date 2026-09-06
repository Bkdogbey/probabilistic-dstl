import pytest
import torch

from pdstl import Always, Eventually, Predicate, Until


def formulas():
    a, b = Predicate("a"), Predicate("b")
    return [a & b, a | b, ~(a & b), Always(a, [1, 3]), Eventually(a, [0, 2]),
            Until(a, b, [0, 3]), Always(Eventually(a & b, [0, 2]), [0, 1]),
            Until(Always(a, [0, 1]), Eventually(b, [0, 1]), [1, 2])]


@pytest.mark.parametrize("scale", [2, 10, 100])
def test_formula_error_bounds(scale):
    generator = torch.Generator().manual_seed(7)
    inputs = {name: torch.rand(2, 8, 2, generator=generator, dtype=torch.float64).sort(-1).values
              for name in ("a", "b")}
    for formula in formulas():
        direct, smooth = formula(inputs), formula(inputs, scale=scale)
        assert (smooth - direct).abs().max() <= formula.smoothing_error(scale) + 1e-12
        assert formula.smoothing_error(-1) == 0


def test_smoothing_is_not_a_probability_interval():
    a = Predicate("a")
    x = torch.full((1, 2, 2), 0.95, dtype=torch.float64)
    formula = Eventually(a, [0, 1])
    assert formula({"a": x}, scale=2).min() > 1
    torch.testing.assert_close(formula({"a": x}), x[:, :1])
    low = torch.full((1, 1, 2), 0.1, dtype=torch.float64)
    result = (a & a)({"a": low}, scale=2)
    assert result[..., 0] > result[..., 1]


def test_smooth_gradients_through_nested_operators():
    generator = torch.Generator().manual_seed(23)
    x = (0.1 + 0.8 * torch.rand(1, 5, 2, generator=generator, dtype=torch.float64)).sort(-1).values.requires_grad_()
    y = (0.1 + 0.8 * torch.rand(1, 5, 2, generator=generator, dtype=torch.float64)).sort(-1).values.requires_grad_()
    a, b = Predicate("a"), Predicate("b")
    formula = Until(Always(~a | b, [0, 1]), Eventually(a & b, [0, 1]), [1, 2])
    assert torch.autograd.gradcheck(lambda p, q: formula({"a": p, "b": q}, scale=5), (x, y))


def test_gradient_reaches_controls_through_supplied_probabilities():
    controls = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    formula = Eventually(Predicate("target"), [1, 3])

    def prediction():
        means = torch.cat([controls.new_zeros(1), controls.cumsum(0)])
        p = 0.5 * (1 + torch.erf((means - 0.5) / 2**0.5))
        return {"target": torch.stack([p, p], -1).unsqueeze(0)}

    initial = formula.robustness(prediction())[0, 0, 0].item()
    optimizer = torch.optim.Adam([controls], lr=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        loss = -formula.robustness(prediction(), scale=10)[..., 0].mean()
        loss.backward()
        assert torch.isfinite(controls.grad).all()
        optimizer.step()
    final = formula.robustness(prediction())[0, 0, 0].item()
    assert final > initial + 0.2
