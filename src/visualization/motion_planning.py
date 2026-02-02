"""
Motion Planning Visualization
==============================
Visualization functions for 2D motion planning trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse


def plot_trajectory_2d(means, vars, obstacle, goal, initial_state, 
                      history=None, title="2D Motion Planning"):
    """
    Plot 2D trajectory with filled uncertainty ellipses and optional metrics.
    
    Args:
        means: [T, 2] trajectory positions
        vars: [T, 2] trajectory variances
        obstacle: dict with 'x' and 'y' bounds
        goal: [2] goal position
        initial_state: [2] start position
        history: list of dicts with 'p_safe', 'p_goal', 'loss' (optional)
        title: plot title
    """
    if history is not None:
        # Two plots: trajectory + metrics
        fig = plt.figure(figsize=(16, 7))
        ax1 = plt.subplot(1, 2, 1)
    else:
        # Single plot: trajectory only
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    
    # =========================================================================
    # LEFT: Trajectory with filled uncertainty ellipses
    # =========================================================================
    
    # Obstacle (FIXED)
    rect = Rectangle(
        (obstacle['x'][0], obstacle['y'][0]),
        obstacle['x'][1] - obstacle['x'][0],
        obstacle['y'][1] - obstacle['y'][0],
        facecolor='red', alpha=0.3, edgecolor='red', linewidth=2, label='Obstacle'
    )
    ax1.add_patch(rect)
    
    # Goal (FIXED)
    goal_circle = Circle(
        goal, 0.5, 
        facecolor='green', alpha=0.2, edgecolor='green', linewidth=2, label='Goal'
    )
    ax1.add_patch(goal_circle)
    ax1.plot(goal[0], goal[1], 'g*', markersize=20, zorder=10)
    
    # Mean trajectory (color-coded by time)
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(means)))
    for i in range(len(means) - 1):
        ax1.plot(means[i:i+2, 0], means[i:i+2, 1], color=colors[i], linewidth=3)
    
    # Start and final markers
    ax1.plot(initial_state[0], initial_state[1], 'ko', markersize=15, 
            label='Start', zorder=10, markeredgewidth=2, markeredgecolor='white')
    ax1.plot(means[-1, 0], means[-1, 1], 'r^', markersize=15,
            label='Final', zorder=10, markeredgewidth=2, markeredgecolor='white')
    
    # Filled uncertainty ellipses (2σ) - sample every few timesteps
    k = 2.0
    for t in range(0, len(means), max(1, len(means)//15)):
        ellipse = Ellipse(
            means[t], 
            2*k*np.sqrt(vars[t, 0]), 
            2*k*np.sqrt(vars[t, 1]),
            facecolor='lightblue', edgecolor='blue', 
            linewidth=1.5, alpha=0.4, linestyle='--'
        )
        ax1.add_patch(ellipse)
    
    ax1.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Trajectory with 2σ Uncertainty', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # =========================================================================
    # RIGHT: Metrics (if history provided)
    # =========================================================================
    
    if history is not None:
        ax2 = plt.subplot(2, 2, 2)  # Top right
        ax3 = plt.subplot(2, 2, 4)  # Bottom right
        
        iterations = range(len(history))
        p_safe = [h['p_safe'] for h in history]
        p_goal = [h['p_goal'] for h in history]
        
        # Top: Probabilities
        ax2.plot(iterations, p_safe, 'b-', linewidth=2.5, label='P_safe')
        ax2.plot(iterations, p_goal, 'g-', linewidth=2.5, label='P_goal')
        ax2.axhline(0.95, color='red', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='Target (0.95)')
        ax2.fill_between(iterations, 0.95, 1.0, alpha=0.15, color='green')
        
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Probability', fontsize=11)
        ax2.set_title('STL Satisfaction', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Bottom: Loss
        if 'loss' in history[0]:
            total_loss = [h['loss'] for h in history]
            ax3.semilogy(iterations, total_loss, 'k-', linewidth=2.5)
            ax3.set_xlabel('Iteration', fontsize=11)
            ax3.set_ylabel('Loss (log scale)', fontsize=11)
            ax3.set_title('Loss Convergence', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig