"""Tests for the pdSTL formula core: Formula, Predicate, Not, And, Or."""

import pytest
import torch

from pdstl import And, Formula, Not, OfflineSource, OnlineSource, Or, Predicate


def _offline(traces):
    return OfflineSource({p: t.unsqueeze(0) for p, t in traces.items()})


def test_predicate_returns_offline_trace_unchanged():
    a = Predicate("A")
    trace = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.1, 0.9]]).unsqueeze(0)
    source = OfflineSource({a: trace})

    assert torch.equal(a(source), trace)


def test_predicate_reads_all_available_online_steps():
    a = Predicate("A")
    source = OnlineSource()
    source.append({a: torch.tensor([[0.25, 0.75]])})
    source.append({a: torch.tensor([[0.4, 0.6]])})

    out = a(source)

    assert out.shape == (1, 2, 2)
    assert out[0, 0].tolist() == pytest.approx([0.25, 0.75])
    assert out[0, 1].tolist() == pytest.approx([0.4, 0.6])


def test_output_shape_is_b_t_2():
    a = Predicate("A")
    b = Predicate("B")
    source = _offline(
        {
            a: torch.tensor([[0.6, 0.9], [0.5, 0.5]]),
            b: torch.tensor([[0.7, 0.95], [0.5, 0.5]]),
        }
    )

    assert a(source).shape == (1, 2, 2)
    assert (a & b)(source).shape == (1, 2, 2)


def test_negation_computes_one_minus_upper_lower():
    a = Predicate("A")
    source = _offline({a: torch.tensor([[0.6, 0.9]])})

    out = Not(a)(source)

    assert out[0, 0].tolist() == pytest.approx([0.1, 0.4])


def test_and_uses_frechet_bounds():
    a = Predicate("A")
    b = Predicate("B")
    source = _offline({a: torch.tensor([[0.6, 0.9]]), b: torch.tensor([[0.7, 0.95]])})

    out = And(a, b)(source)

    assert out[0, 0].tolist() == pytest.approx([0.3, 0.9])


def test_and_does_not_use_the_product_lower_bound():
    a = Predicate("A")
    b = Predicate("B")
    source = _offline({a: torch.tensor([[0.6, 0.9]]), b: torch.tensor([[0.7, 0.95]])})

    out = And(a, b)(source)

    product_lower = 0.6 * 0.7
    frechet_lower = 0.3
    assert out[0, 0, 0].item() == pytest.approx(frechet_lower)
    assert out[0, 0, 0].item() != pytest.approx(product_lower)


def test_or_uses_frechet_bounds():
    a = Predicate("A")
    b = Predicate("B")
    source = _offline({a: torch.tensor([[0.6, 0.9]]), b: torch.tensor([[0.7, 0.95]])})

    out = Or(a, b)(source)

    assert out[0, 0].tolist() == pytest.approx([0.7, 1.0])


def test_and_of_identical_predicate_returns_its_own_bounds():
    a = Predicate("A")
    source = _offline({a: torch.tensor([[0.6, 0.9]])})

    out = (a & a)(source)

    assert out[0, 0].tolist() == pytest.approx([0.6, 0.9])


def test_or_of_identical_predicate_returns_its_own_bounds():
    a = Predicate("A")
    source = _offline({a: torch.tensor([[0.6, 0.9]])})

    out = (a | a)(source)

    assert out[0, 0].tolist() == pytest.approx([0.6, 0.9])


def test_nested_boolean_formulas():
    a = Predicate("A")
    b = Predicate("B")
    c = Predicate("C")
    source = _offline(
        {
            a: torch.tensor([[0.6, 0.9]]),
            b: torch.tensor([[0.7, 0.95]]),
            c: torch.tensor([[0.4, 0.7]]),
        }
    )

    formula = (a & b) | ~c
    out = formula(source)

    and_bounds = torch.tensor([0.3, 0.9])
    not_c = torch.tensor([0.3, 0.6])
    expected_lower = max(and_bounds[0], not_c[0])
    expected_upper = min(1.0, (and_bounds[1] + not_c[1]).item())
    assert out[0, 0].tolist() == pytest.approx([expected_lower, expected_upper])


def test_operator_overloads_construct_expected_types():
    a = Predicate("A")
    b = Predicate("B")

    assert isinstance(a & b, And)
    assert isinstance(a | b, Or)
    assert isinstance(~a, Not)


def test_child_formulas_are_registered_as_submodules():
    a = Predicate("A")
    b = Predicate("B")

    conjunction = a & b
    children = dict(conjunction.named_children())
    assert children == {"left": a, "right": b}

    negation = ~a
    assert dict(negation.named_children()) == {"child": a}


def test_batched_traces_are_supported():
    a = Predicate("A")
    b = Predicate("B")
    source = OfflineSource(
        {
            a: torch.tensor([[0.6, 0.9]] * 3).unsqueeze(1),
            b: torch.tensor([[0.7, 0.95]] * 3).unsqueeze(1),
        }
    )

    out = (a & b)(source)

    assert out.shape == (3, 1, 2)
    assert torch.allclose(out, torch.tensor([[0.3, 0.9]] * 3).unsqueeze(1))


def test_gradients_propagate_from_lower_bound_to_source_tensors():
    a = Predicate("A")
    b = Predicate("B")
    a_trace = torch.tensor([[[0.6, 0.9]]], requires_grad=True)
    b_trace = torch.tensor([[[0.7, 0.95]]], requires_grad=True)
    source = OfflineSource({a: a_trace, b: b_trace})

    out = (a & b)(source)
    out[0, 0, 0].backward()

    assert a_trace.grad is not None
    assert b_trace.grad is not None
    assert torch.any(a_trace.grad != 0)
    assert torch.any(b_trace.grad != 0)


def test_forward_is_not_implemented_on_bare_formula():
    class Bare(Formula):
        pass

    with pytest.raises(NotImplementedError):
        Bare().forward(None)
