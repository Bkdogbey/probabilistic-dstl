"""Print deterministic numerical evidence used by PDSTL_AUDIT_REPORT.md."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import multivariate_normal, norm

from conftest import ConstantFormula, exact_until
from evaluation.monte_carlo import rollout_controls
from models.dynamics import GaussianBelief
from pdstl.base import BeliefTrajectory
from pdstl.operators import Always, Eventually, GreaterThan, LessThan, Until
from planning.dynamics import SingleIntegrator
from planning.environment import RectangularGoalPredicate, RectangularObstaclePredicate, normal_cdf
from planning.planner import TorchGaussianBelief


def scalar_belief(mean, variance, confidence=2.0):
    return GaussianBelief(
        torch.tensor([[[mean]]], dtype=torch.float64),
        torch.tensor([[[variance]]], dtype=torch.float64),
        confidence_level=confidence,
    )


def scalar_value(predicate, mean, variance, confidence=2.0):
    value = predicate(BeliefTrajectory([scalar_belief(mean, variance, confidence)]))
    return [float(item) for item in value[0, 0]]


def torch_belief(covariance):
    return TorchGaussianBelief(
        torch.zeros(1, 2, dtype=torch.float64),
        torch.as_tensor(covariance, dtype=torch.float64).reshape(1, 2, 2),
    )


def main():
    data = {}
    data["atomic"] = {
        "ge_N0_1": {"observed": scalar_value(GreaterThan(0), 0, 1), "oracle": 0.5},
        "ge_N1_4": {
            "observed": scalar_value(GreaterThan(0), 1, 4),
            "oracle": float(norm.cdf(0.5)),
        },
        "le_N0_1": {"observed": scalar_value(LessThan(0), 0, 1), "oracle": 0.5},
        "le_N1_4": {
            "observed": scalar_value(LessThan(0), 1, 4),
            "oracle": float(norm.cdf(-0.5)),
        },
        "confidence_1_ge_N0_1": scalar_value(GreaterThan(0), 0, 1, 1),
    }

    means = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    data["zero_variance_ge"] = [
        float(item) for item in 1.0 - normal_cdf(0.0, means, torch.zeros_like(means))
    ]
    data["zero_variance_scalar_ge"] = {
        str(mean): scalar_value(GreaterThan(0), mean, 0.0)
        for mean in [-1.0, 0.0, 1.0]
    }
    tiny = {}
    for variance in [1e-16, 1e-12, 1e-8]:
        mean = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        value = 1.0 - normal_cdf(0.0, mean, torch.tensor(variance, dtype=torch.float64))
        value.backward()
        tiny[f"{variance:.0e}"] = {"p_at_equality": value.item(), "dp_dmean": mean.grad.item()}
    data["tiny_variance"] = tiny

    region = {"x": [-1, 1], "y": [-1, 1]}
    axis = float(norm.cdf(1) - norm.cdf(-1))
    independent = BeliefTrajectory([torch_belief(np.eye(2))])
    correlated_cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    correlated = BeliefTrajectory([torch_belief(correlated_cov)])
    true_correlated = float(
        multivariate_normal.cdf(
            [1, 1], mean=[0, 0], cov=correlated_cov, lower_limit=[-1, -1],
            rng=np.random.default_rng(20260814),
        )
    )
    data["rectangles"] = {
        "axis_probability": axis,
        "independent_inside_oracle": axis**2,
        "shared_goal_independent": float(RectangularGoalPredicate(region)(independent)[0, 0, 0]),
        "outside_oracle": 1.0 - axis**2,
        "shared_obstacle_independent": float(RectangularObstaclePredicate(region)(independent)[0, 0, 0]),
        "correlated_inside_scipy": true_correlated,
        "shared_goal_correlated": float(RectangularGoalPredicate(region)(correlated)[0, 0, 0]),
        "frechet": [max(0.0, 2.0 * axis - 1.0), axis],
    }

    phi = np.array(
        [[0.85, 0.95], [0.55, 0.80], [0.70, 0.90], [0.25, 0.60], [0.65, 0.75]]
    )
    psi = np.array(
        [[0.20, 0.45], [0.75, 0.90], [0.35, 0.55], [0.80, 0.95], [0.50, 0.85]]
    )
    until_rows = {}
    for interval in [[0, 1], [0, 2], [1, 2], [0, np.inf]]:
        key = str(interval)
        observed = Until(ConstantFormula(phi), ConstantFormula(psi), interval=interval)(None)[0]
        expected = exact_until(phi, psi, interval)
        until_rows[key] = {
            "observed_t0": observed[0].tolist(),
            "oracle_t0": expected[0].tolist(),
        }
    single = Until(ConstantFormula([[0.6, 0.6]]), ConstantFormula([[0.7, 0.7]]), [0, 0])(None)
    data["until"] = {"single_observed": single[0, 0].tolist(), "single_oracle": [0.3, 0.6], **until_rows}

    smooth = {}
    formulas = {
        "always": Always(ConstantFormula(phi), interval=[0, 4]),
        "eventually": Eventually(ConstantFormula(phi), interval=[0, 4]),
        "until": Until(ConstantFormula(phi), ConstantFormula(psi), interval=[0, 4]),
    }
    for name, formula in formulas.items():
        exact = formula(None, scale=-1)
        smooth[name] = {
            str(beta): float(torch.max(torch.abs(formula(None, scale=beta) - exact)))
            for beta in [1, 5, 10, 50, 100, 500]
        }
    smooth["explicit"] = {
        "min_005_beta10": float(Always(ConstantFormula([[0.05, 0.05]] * 3), [0, 2])(None, scale=10)[0, 0, 0]),
        "max_099_beta10": float(Eventually(ConstantFormula([[0.99, 0.99]] * 3), [0, 2])(None, scale=10)[0, 0, 0]),
        "max_099_beta50": float(Eventually(ConstantFormula([[0.99, 0.99]] * 3), [0, 2])(None, scale=50)[0, 0, 0]),
    }
    data["smoothing"] = smooth

    gradients = {}
    for score in [-0.05, 0.2, 1.05]:
        value = torch.tensor(score, requires_grad=True)
        loss = -torch.log(torch.clamp(value, min=1e-6, max=1.0))
        loss.backward()
        gradients[str(score)] = {"loss": loss.item(), "gradient": value.grad.item()}
    data["clamp_gradients"] = gradients

    rollouts = rollout_controls(
        SingleIntegrator(q_std=0.05), torch.zeros(2), torch.eye(2) * 0.25,
        torch.zeros(1, 2), num_rollouts=32, seed=140826,
    )
    data["monte_carlo_initial"] = {
        "requested_covariance_diagonal": [0.25, 0.25],
        "sample_variance_diagonal": torch.var(rollouts[:, 0, :], dim=0).tolist(),
        "unique_initial_states": int(torch.unique(rollouts[:, 0, :], dim=0).shape[0]),
    }

    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
