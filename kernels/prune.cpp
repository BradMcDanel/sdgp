#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor prune_cuda(const at::Tensor x,
                      const int prune_type,
                      const int nonzero,
                      const int group_size);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

at::Tensor prune(const at::Tensor x,
                 const int prune_type,
                 const int nonzero,
                 const int group_size) {
  return prune_cuda(x, prune_type, nonzero, group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prune", &prune, "Prune (CUDA)");
}
