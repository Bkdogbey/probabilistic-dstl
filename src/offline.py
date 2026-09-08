"""Shared execution pipeline for offline pdSTL examples."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from models.dynamics import (
    create_gaussian_belief_trajectory,
    create_signal_trace,
)
from pdstl.operators import Always, Eventually, GreaterThan, STL_Formula
from utils import to_steps
from visualization.temporal import present_temporal_example


@dataclass
class OfflineResult:
    """Values produced by one offline example."""

    title: str
    signal_type: str
    time: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    sigma_multiplier: float
    predicate: GreaterThan
    formula: STL_Formula
    formula_config: dict
    atomic_trace: torch.Tensor
    temporal_trace: torch.Tensor
    inner_formula: Optional[STL_Formula] = None
    inner_trace: Optional[torch.Tensor] = None


def _build_formula(config, predicate, time):
    """Build an Always/Eventually formula from validated identifiers."""
    child_config = config.get("child")
    child = (
        predicate
        if child_config is None
        else _build_formula(child_config, predicate, time)
    )

    if "interval_sec" in config:
        interval = to_steps(config["interval_sec"], time)
    elif "interval_steps" in config:
        interval = config["interval_steps"]
    else:
        raise ValueError("formula requires interval_sec or interval_steps")

    operators = {"always": Always, "eventually": Eventually}
    operator = config.get("operator")
    if operator not in operators:
        raise ValueError("formula operator must be 'always' or 'eventually'")
    return operators[operator](child, interval=interval)


def evaluate_offline_example(title, config):
    """Evaluate one configured signal and temporal formula."""
    time, mean, variance = create_signal_trace(config["signal"])
    beliefs = create_gaussian_belief_trajectory(
        mean,
        variance,
        sigma_multiplier=config["sigma_multiplier"],
        dtype=torch.float64,
    )
    predicate = GreaterThan(config["threshold"])
    formula = _build_formula(config["formula"], predicate, time)
    atomic_trace = predicate(beliefs)

    inner_formula = None
    inner_trace = None
    if not formula.subformula.is_pointwise:
        inner_formula = formula.subformula
        inner_trace = inner_formula(beliefs, scale=-1)

    return OfflineResult(
        title=title,
        signal_type=config["signal"]["type"],
        time=time,
        mean=mean,
        variance=variance,
        sigma_multiplier=config["sigma_multiplier"],
        predicate=predicate,
        formula=formula,
        formula_config=config["formula"],
        atomic_trace=atomic_trace,
        inner_formula=inner_formula,
        inner_trace=inner_trace,
        temporal_trace=formula(beliefs, scale=-1),
    )


def run_offline_example(title, config, show=True):
    """Evaluate and present one offline example."""
    result = evaluate_offline_example(title, config)
    present_temporal_example(result, show=show)
    return result
