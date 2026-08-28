"""Smoke tests for the installed pdstl package: import, public API, and
absence of the removed compiled/recurrent surface.
"""

from pathlib import Path

import torch


def test_import_succeeds():
    import pdstl  # noqa: F401


def test_public_symbols_are_importable():
    from pdstl import (  # noqa: F401
        Always,
        And,
        Eventually,
        Formula,
        Not,
        OfflineSource,
        OnlineSource,
        Or,
        Predicate,
        ProbabilitySource,
        TemporalOperator,
        validate_bounds,
    )


def test_basic_offline_formula_runs():
    from pdstl import Always, OfflineSource, Predicate

    a = Predicate("A")
    b = Predicate("B")
    source = OfflineSource(
        {
            a: torch.tensor([[0.6, 0.9], [0.6, 0.9]]).unsqueeze(0),
            b: torch.tensor([[0.7, 0.95], [0.7, 0.95]]).unsqueeze(0),
        }
    )

    out = Always(a & b, (0, 1))(source)

    assert out.shape == (1, 1, 2)


def test_public_api_omits_obsolete_symbols():
    import pdstl

    obsolete = [
        "STLFormula",
        "STL_Formula",
        "Negation",
        "Until",
        "compile_formula",
        "compile_recurrent_formula",
        "TensorProbabilitySource",
        "TableProbabilitySource",
        "BeliefTrajectory",
        "CompiledFormula",
        "RecurrentFormula",
        "evaluate",
    ]
    for name in obsolete:
        assert not hasattr(pdstl, name), f"obsolete symbol still exported: {name}"


def test_installed_package_exposes_only_current_modules():
    import pdstl

    package_dir = Path(pdstl.__file__).parent
    modules = {p.stem for p in package_dir.glob("*.py")}

    assert modules == {"__init__", "base", "operators"}
