import pytest
import torch

from pdstl import Always, Eventually, Predicate, Until


def trace(seed=1, length=9, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, length, 2, generator=generator, dtype=dtype).sort(-1).values


@pytest.mark.parametrize("kind,reduce", [(Always, torch.amin), (Eventually, torch.amax)])
@pytest.mark.parametrize("interval", [(0, 0), (0, 3), (1, 3), (3, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrix_windows_match_direct_slices(kind, reduce, interval, dtype):
    x = trace(dtype=dtype)
    formula = kind(Predicate("a"), interval)
    a, b = interval
    expected = torch.stack([reduce(x[:, t+a:t+b+1], dim=1)
                            for t in range(x.shape[1] - b)], dim=1)
    actual = formula({"a": x})
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype
    assert {"M", "b"} <= dict(formula.named_buffers()).keys()
    formula.double()
    assert formula.M.dtype == torch.float64


@pytest.mark.parametrize("interval", [(0, 0), (0, 3), (1, 3), (3, 3)])
def test_until_matches_independent_enumeration(interval):
    left, right = trace(1), trace(2)
    a, b = interval
    # A slow scalar reference enumerates every witness and inclusive prefix.
    expected = []
    for batch in range(left.shape[0]):
        outputs = []
        for t in range(left.shape[1] - b):
            candidates = []
            for tau in range(t + a, t + b + 1):
                lo = min(float(left[batch, s, 0]) for s in range(t, tau + 1))
                hi = min(float(left[batch, s, 1]) for s in range(t, tau + 1))
                candidates.append((max(0, lo + float(right[batch, tau, 0]) - 1),
                                   min(hi, float(right[batch, tau, 1]))))
            outputs.append([max(pair[0] for pair in candidates),
                            max(pair[1] for pair in candidates)])
        expected.append(outputs)
    formula = Until(Predicate("a"), Predicate("b"), interval)
    torch.testing.assert_close(formula({"a": left, "b": right}),
                               torch.tensor(expected, dtype=left.dtype))


def test_nested_windows_align_at_same_origins():
    x = trace()
    y = trace(2)
    a, b = Predicate("a"), Predicate("b")
    formula = Always(Eventually(a, [1, 2]), [0, 2]) | b
    expected = []
    for t in range(x.shape[1] - 4):
        inner = torch.stack([x[:, s+1:s+3].amax(1) for s in range(t, t+3)], 1).amin(1)
        expected.append(torch.stack([torch.maximum(inner[:, 0], y[:, t, 0]),
                                     (inner[:, 1] + y[:, t, 1]).clamp(max=1)], -1))
    torch.testing.assert_close(formula({"a": x, "b": y}), torch.stack(expected, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_windows_and_gradients():
    x = trace().cuda().requires_grad_()
    formula = Always(Predicate("a"), [1, 3]).cuda()
    output = formula({"a": x}, scale=5)
    output.sum().backward()
    assert output.is_cuda and torch.isfinite(x.grad).all()
