# Adapted from KernelBench level1/5_Matrix_scalar_multiplication.py (https://github.com/ScalingIntelligence/KernelBench).
# Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini, Scaling Intelligence Lab, Stanford University.
# MIT License, see LICENSE in this directory. Modified: shapes reduced so one evaluation takes seconds.
# The evaluator does not use these torch.rand draws for correctness; it redraws signed torch.randn inputs of the same shapes.
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return A * s

M = 1024
N = 2048

def get_inputs():
    A = torch.rand(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed
