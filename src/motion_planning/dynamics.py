"""
2D Stochastic Dynamics for Motion Planning
===========================================
"""

import torch


class SingleIntegrator2D:
    """
    2D stochastic single-integrator dynamics: ẋ = u + w
    
    State: x = [x, y] (position)
    Control: u = [u_x, u_y] (velocity commands)
    Noise: w ~ N(0, σ²I)
    
    Args:
        dt: Time step (seconds)
        sigma: Process noise standard deviation
    """
    
    def __init__(self, dt=0.1, sigma=0.1):
        self.dt = dt
        self.sigma = sigma
        self.Q = (sigma**2) * dt * torch.eye(2)
    
    def step(self, mu, var, u):
        """
        Propagate Gaussian belief one time step.
        
        Args:
            mu: [B, 1, 2] mean position
            var: [B, 1, 2] variance (diagonal)
            u: [B, 1, 2] control input
            
        Returns:
            mu_next: [B, 1, 2] updated mean
            var_next: [B, 1, 2] updated variance
        """
        mu_next = mu + self.dt * u
        var_next = var + self.Q.diagonal().view(1, 1, 2)
        return mu_next, var_next
    
    def __str__(self):
        return f"SingleIntegrator2D(dt={self.dt}, σ={self.sigma})"