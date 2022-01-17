import torch

from torch.utils.cpp_extension import load

prune = load(name="prune", sources=["kernels/prune.cpp", "kernels/prune_kernel.cu"],
             verbose=False)

x = torch.randn(8, 4, 1, 1).cuda()

# y = prune.prune(x, 1, 0, 2, 8)
# print(x[:, -1, -1, -1])
# print(y[:, -1, -1, -1])
# print(y)

ys = []
for i in range(1000):
    y = prune.prune(x, 1, 1, 2, 4)
    y = (y[-1, :, -1, -1] == 0)
    ys.append(y.cpu().numpy().tolist())

ys = torch.tensor(ys).cuda()
print(ys.sum(dim=0))
