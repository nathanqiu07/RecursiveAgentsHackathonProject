# Adapted from KernelBench level1/19_ReLU.py (https://github.com/ScalingIntelligence/KernelBench).
# Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini, Scaling Intelligence Lab, Stanford University.
# MIT License, see LICENSE in this directory. Modified: shapes reduced so one evaluation takes seconds.
# The evaluator does not use these torch.rand draws for correctness; it redraws signed torch.randn inputs of the same shapes.
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return torch.relu(x)

batch_size = 64
dim = 16384

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
