import torch
import torch.nn as nn
import numpy as np

class STL_Formula(nn.Module):
    """Base class for all STL formulas."""
    
    def __init__(self):
        super(STL_Formula, self).__init__()
    
    def forward(self, trace, scale=-1):
        return self.robustness_trace(trace, scale=scale)
    
    def robustness_trace(self, trace, scale=-1):
        raise NotImplementedError()

class GreaterThan(STL_Formula):
    """Atomic predicate: x >= threshold"""
    
    def __init__(self, threshold):
        super(GreaterThan, self).__init__()
        self.threshold = threshold
    
    def robustness_trace(self, trace, scale=-1):
        return trace - self.threshold
    
    def __str__(self):
        return f"x >= {self.threshold}"

class Always(STL_Formula):
    """Always operator: G[a,b](φ) - simplified version"""
    
    def __init__(self, subformula, interval=None):
        super(Always, self).__init__()
        self.subformula = subformula
        self.interval = interval if interval is not None else [0, np.inf]
    
    def forward(self, trace, scale=-1):
        """
        Simplified Always operator using sliding window min
        trace: (batch, time, features) - can be original time order
        Returns: (batch, time, features) - robustness in original time order
        """
        # Get subformula robustness
        sub_robustness = self.subformula(trace, scale=scale)
        
        batch_size, time_steps, features = sub_robustness.shape
        
        if self.interval[1] == np.inf:
            # Unbounded: cumulative minimum
            result = torch.zeros_like(sub_robustness)
            for i in range(time_steps):
                result[:, i, :] = torch.min(sub_robustness[:, :i+1, :], dim=1)[0]
        else:
            # Bounded: sliding window minimum
            a, b = int(self.interval[0]), int(self.interval[1])
            window_size = b - a + 1
            result = torch.zeros_like(sub_robustness)
            
            for i in range(time_steps):
                start = max(0, i - b)
                end = i - a + 1
                if end > start:
                    result[:, i, :] = torch.min(sub_robustness[:, start:end, :], dim=1)[0]
                else:
                    # Not enough points in window, use available points
                    result[:, i, :] = torch.min(sub_robustness[:, :i+1, :], dim=1)[0]
        
        return result
    
    def __str__(self):
        return f"G{self.interval}({self.subformula})"