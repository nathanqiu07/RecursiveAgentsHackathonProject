# Adapted from KernelBench level1/36_RMSNorm_.py (https://github.com/ScalingIntelligence/KernelBench).
# Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini, Scaling Intelligence Lab, Stanford University.
# MIT License, see LICENSE in this directory. Modified: shapes reduced so one evaluation takes seconds.
# The evaluator does not use these torch.rand draws for correctness; it redraws signed torch.randn inputs of the same shapes.
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms

batch_size = 8
features = 64
dim1 = 64
dim2 = 64

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
