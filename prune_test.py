import torch

from torch.utils.cpp_extension import load
import gsr

prune = load(name="prune", sources=["kernels/prune.cpp", "kernels/prune_kernel.cu"],
             verbose=False)

x = torch.randn(4, 8, 1, 1).cuda()

y = prune.prune(x, gsr.PRUNE_TYPE_STC, 2, 4)
print(x[-1, :4, -1, -1])
print(y[-1, :4, -1, -1])
# print(y)

assert False
ys = []
for i in range(1000):
    y = prune.prune(x, gsr.PRUNE_TYPE_RND, 2, 4)
    y = (y[-1, :, -1, -1] == 0)
    ys.append(y.cpu().numpy().tolist())

ys = torch.tensor(ys).cuda()
print(ys.sum(dim=0))
