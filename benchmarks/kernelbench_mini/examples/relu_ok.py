"""A known-good ModelNew for 19_ReLU: the GPU smoke test and the shape every candidate must take."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    long n = x.numel();
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

cpp_src = "torch::Tensor relu_cuda(torch::Tensor x);"

relu_ext = load_inline(name="kbmini_relu_ok", cpp_sources=cpp_src, cuda_sources=cuda_src,
                       functions=["relu_cuda"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return relu_ext.relu_cuda(x.contiguous())
