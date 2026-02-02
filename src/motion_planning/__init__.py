"""
Motion Planning Module
=======================
2D motion planning with probabilistic STL specifications.

Provides:
- Spatial predicates (obstacle avoidance, goal reaching)
- 2D stochastic dynamics
- Gradient-based trajectory optimization
"""

from motion_planning.predicates import ObstacleAvoidance, GoalReaching
from motion_planning.dynamics import SingleIntegrator2D
from motion_planning.planning_2d import optimize_trajectory

__all__ = [
    'ObstacleAvoidance', 
    'GoalReaching', 
    'SingleIntegrator2D',
    'optimize_trajectory'
]
