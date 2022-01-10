import torch

from torch.utils.cpp_extension import load

prune = load(name="prune", sources=["kernels/prune.cpp", "kernels/prune_kernel.cu"],
             verbose=False)



x = torch.randn(4, 8, 2, 2).cuda()
y = prune.prune(x, 0, 4, 8)

print(x[0, :8, 0, 0], y[0, :8, 0, 0])
