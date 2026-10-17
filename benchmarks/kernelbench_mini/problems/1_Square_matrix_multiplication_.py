# Adapted from KernelBench level1/1_Square_matrix_multiplication_.py (https://github.com/ScalingIntelligence/KernelBench).
# Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini, Scaling Intelligence Lab, Stanford University.
# MIT License, see LICENSE in this directory. Modified: shapes reduced so one evaluation takes seconds.
# The evaluator does not use these torch.rand draws for correctness; it redraws signed torch.randn inputs of the same shapes.
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)

N = 256

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
