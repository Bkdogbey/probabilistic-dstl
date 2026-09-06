"""End-to-end checks that the existing call sites still work after the input
contract change: the two main.py demos, the visualization adapter, the
deterministic baseline, and a real planner optimisation."""

import numpy as np
import torch

from baselines.det_stl import DetAlways, DetRectangularObstaclePredicate
from models.dynamics import (
    GaussianBelief,
    linear_system,
    piecewise_signal,
    sinusoidial_input,
)
from pdstl.base import BeliefTrajectory, OnlineBeliefTrajectory
from pdstl.operators import Always, GreaterThan
from planning.dynamics import SingleIntegrator
from planning.environment import Environment
from planning.planner import Planner, TorchGaussianBelief
from utils import create_belief_trajectory, load_config, to_steps
from visualization.robustness import _to_numpy


def test_main_example1_linear_system_path():
    d = load_config("configs/stl_demos.yaml")["example1"]
    t = np.linspace(0, d["t_end"], d["n_steps"])
    mean, var = linear_system(
        a=d["a"], b=d["b"], g=d["g"], q=d["q"],
        mu=d["mu"], P=d["P"], t=t, control_func=sinusoidial_input,
    )

    beliefs = create_belief_trajectory(mean, var)
    phi = GreaterThan(d["threshold"])
    spec = Always(phi, interval=to_steps(d["interval_sec"], t))

    assert phi(beliefs).shape == (1, d["n_steps"], 2)
    assert spec(beliefs).shape == (1, d["n_steps"], 2)


def test_main_example2_piecewise_path_and_visualization_adapter():
    t, mean, var = piecewise_signal()
    beliefs = create_belief_trajectory(mean, var)
    phi = GreaterThan(50.0)

    trace = Always(phi, interval=[1, 2])(beliefs)

    # the plotting helpers consume the trace unchanged
    assert _to_numpy(trace, len(t)).shape == (len(t), 2)
    assert _to_numpy(phi(beliefs), len(t)).shape == (len(t), 2)


def test_deterministic_baseline_reads_the_belief_through_value():
    mean = torch.randn(1, 6, 2)
    traj = BeliefTrajectory(
        [TorchGaussianBelief(mean[:, i, :], torch.eye(2) * 0.1) for i in range(6)]
    )
    spec = DetAlways(
        DetRectangularObstaclePredicate({"x": [2.0, 3.0], "y": [2.0, 3.0]}),
        interval=[0, 3],
    )

    assert spec(traj).shape == (1, 6, 1)


def test_planner_optimisation_runs_and_improves():
    torch.manual_seed(0)
    env = Environment(device="cpu")
    env.set_goal(x_range=[1.0, 1.8], y_range=[-0.5, 0.5])
    env.add_obstacle(x_range=[0.4, 0.7], y_range=[0.6, 1.0])

    dyn = SingleIntegrator(dt=0.2, u_max=1.0, q_std=0.05, device="cpu")
    planner = Planner(dyn, env, T=8, config={"max_iters": 40, "min_iters": 10})

    mean_trace, _cov, _u, best_p, history = planner._optimize_window(
        torch.tensor([0.0, 0.0]), torch.eye(2) * 0.01, verbose=False
    )

    assert mean_trace.shape == (1, 9, 2)
    assert 0.0 <= best_p <= 1.0
    assert history[-1] < history[0]  # the objective actually moved


def test_planner_belief_also_answers_comparison_predicates():
    """TorchGaussianBelief used to raise here; it now returns the exact
    probability as equal endpoints, matching the planning predicates."""
    traj = BeliefTrajectory(
        [TorchGaussianBelief(
            torch.tensor([[1.0, 2.0]]), (torch.eye(2) * 0.25).unsqueeze(0)
        )]
    )

    trace = GreaterThan(1.0, dim=0)(traj)

    np.testing.assert_allclose(trace[0, 0].detach().numpy(), [0.5, 0.5], atol=1e-6)


def test_streaming_trajectory_can_be_evaluated_as_it_grows():
    online = OnlineBeliefTrajectory()
    spec = Always(GreaterThan(50.0), interval=[0, 1])

    shapes = []
    for i in range(5):
        online.append(
            GaussianBelief(
                torch.tensor([[50.0 + i]]), torch.tensor([[4.0]]), confidence_level=1.0
            )
        )
        shapes.append(tuple(spec(online).shape))

    assert shapes == [(1, n + 1, 2) for n in range(5)]
