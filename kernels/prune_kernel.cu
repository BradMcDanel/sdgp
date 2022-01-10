#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define MAX_GROUP_SIZE 32

#define PRUNE_MAX 0

#define idx_4d(b, c, w, h, C, W, H) ((b) * (C) * (H) * (W) + (c) * (H) * (W) + (w) * (H) + (h))
#define get_0dim(idx, C, W, H) (idx / ((C) * (W) * (H)))
#define get_1dim(idx, C, W, H) ((idx / (W) / (H)) % (C))
#define get_2dim(idx, C, W, H) ((idx / (H)) % (W))
#define get_3dim(idx, C, W, H) (idx % (H))



namespace {


// prune_max_channelwise
template<typename scalar_t>
__device__ void prune_max_channelwise(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t prune_type,
    const int32_t nonzero,
    const int32_t group_size,
    const int32_t B,
    const int32_t C,
    const int32_t W,
    const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t b = get_0dim(idx, C, W, H);
  const int32_t c = get_1dim(idx, C, W, H);
  const int32_t w = get_2dim(idx, C, W, H);
  const int32_t h = get_3dim(idx, C, W, H);
  
  int32_t heap[MAX_GROUP_SIZE];

  for (int32_t i = 0; i < group_size; i++) {
    int32_t ci = c*group_size + i;
    heap[i] = idx_4d(b, ci, w, h, C, W, H);
  }

  // sort heap (using bubble sort)
  for (int32_t i = 0; i < group_size; i++) {
    for (int32_t j = 0; j < group_size - i - 1; j++) {
      if (abs(x[heap[j]]) < abs(x[heap[j+1]])) {
        int32_t tmp = heap[j];
        heap[j] = heap[j+1];
        heap[j+1] = tmp;
      }
    }
  }

  // assign nonzero elements to y
  for (int32_t i = 0; i < nonzero; i++) {
    y[heap[i]] = x[heap[i]];
  }
}

template <typename scalar_t>
__global__ void prune_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t prune_type,
    const int32_t nonzero,
    const int32_t group_size,
    const int32_t B,
    const int32_t C,
    const int32_t W,
    const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t c = get_1dim(idx, C, W, H);

  if (c >= C / group_size || idx >= B*W*H*C) {
    return;
  }

  // keep largest nonzero elements in each group (across C dimension)
  if (prune_type == PRUNE_MAX) {
    prune_max_channelwise<scalar_t>(
        x, y, prune_type, nonzero, group_size, B, C, W, H);
  }
}
} // namespace

at::Tensor prune_cuda(const at::Tensor x, const int prune_type,
                      const int nonzero, const int group_size) {
  const auto B = x.size(0);
  const auto C = x.size(1);
  const auto W = x.size(2);
  const auto H = x.size(3);
  const auto size = B*C*W*H;
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  auto y = at::zeros_like(x);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "prune_cuda", ([&] {
    prune_kernel<scalar_t><<<blocks, threads>>>(
        x.data<scalar_t>(),
        y.data<scalar_t>(),
        prune_type,
        nonzero,
        group_size,
        B,
        C,
        W,
        H);
  }));

  return y;
}
