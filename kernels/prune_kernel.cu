#include <ATen/ATen.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
// #include <ATen/cuda/detail/PhiloxCudaStateRaw.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <iostream>
#include <vector>

#define MAX_GROUP_SIZE 32

#define PRUNE_MAX 0
#define PRUNE_RND 1

//#define UINT_MAX (__INT_MAX__ * 2U + 1)
#define UNROLL 4
#define idx_4d(b, c, w, h, C, W, H)                                            \
  ((b) * (C) * (H) * (W) + (c) * (H) * (W) + (w) * (H) + (h))
#define get_0dim(idx, C, W, H) (idx / ((C) * (W) * (H)))
#define get_1dim(idx, C, W, H) ((idx / (W) / (H)) % (C))
#define get_2dim(idx, C, W, H) ((idx / (H)) % (W))
#define get_3dim(idx, C, W, H) (idx % (H))

namespace {


template <typename scalar_t>
__device__ void abs_bubble_sort(const scalar_t *__restrict__ vals,
                                int32_t *idxs, const int32_t size) {
  for (int32_t i = 0; i < size; i++) {
    for (int32_t j = 0; j < size - i - 1; j++) {
      if (abs(vals[idxs[j]]) < abs(vals[idxs[j + 1]])) {
        int32_t tmp = idxs[j];
        idxs[j] = idxs[j + 1];
        idxs[j + 1] = tmp;
      }
    }
  }
}

// prune_max_channelwise
template <typename scalar_t>
__device__ void
prune_max_channelwise(const scalar_t *__restrict__ x, scalar_t *__restrict__ y,
                      const int32_t nonzero, const int32_t group_size,
                      const int32_t B, const int32_t C, const int32_t W,
                      const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t b = get_0dim(idx, C, W, H);
  const int32_t c = get_1dim(idx, C, W, H);
  const int32_t w = get_2dim(idx, C, W, H);
  const int32_t h = get_3dim(idx, C, W, H);

  if (c >= C / group_size) {
    return;
  }

  int32_t heap[MAX_GROUP_SIZE];

  for (int32_t i = 0; i < group_size; i++) {
    int32_t ci = c * group_size + i;
    heap[i] = idx_4d(b, ci, w, h, C, W, H);
  }

  abs_bubble_sort<scalar_t>(x, heap, group_size);

  // assign nonzero elements to y
  for (int32_t i = 0; i < nonzero; i++) {
    y[heap[i]] = x[heap[i]];
  }
}

// prune_max_batchwise
template <typename scalar_t>
__device__ void
prune_max_batchwise(const scalar_t *__restrict__ x, scalar_t *__restrict__ y,
                    const int32_t nonzero, const int32_t group_size,
                    const int32_t B, const int32_t C, const int32_t W,
                    const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t b = get_0dim(idx, C, W, H);
  const int32_t c = get_1dim(idx, C, W, H);
  const int32_t w = get_2dim(idx, C, W, H);
  const int32_t h = get_3dim(idx, C, W, H);

  if (b >= B / group_size) {
    return;
  }

  int32_t heap[MAX_GROUP_SIZE];

  for (int32_t i = 0; i < group_size; i++) {
    int32_t bi = b * group_size + i;
    heap[i] = idx_4d(bi, c, w, h, C, W, H);
  }

  abs_bubble_sort<scalar_t>(x, heap, group_size);

  // assign nonzero elements to y
  for (int32_t i = 0; i < nonzero; i++) {
    y[heap[i]] = x[heap[i]];
  }
}

template <typename scalar_t>
__device__ void
prune_rnd_batchwise(const scalar_t *__restrict__ x, scalar_t *__restrict__ y,
                    curandStatePhilox4_32_10_t *state,
                    const int32_t nonzero, const int32_t group_size,
                    const int32_t B, const int32_t C, const int32_t W,
                    const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t b = get_0dim(idx, C, W, H);
  const int32_t c = get_1dim(idx, C, W, H);
  const int32_t w = get_2dim(idx, C, W, H);
  const int32_t h = get_3dim(idx, C, W, H);

  if (b >= B / group_size) {
    return;
  }

  int32_t heap[MAX_GROUP_SIZE];

  for (int32_t i = 0; i < group_size; i++) {
    int32_t bi = b * group_size + i;
    heap[i] = idx_4d(bi, c, w, h, C, W, H);
  }

  // Randomly shuffle the heap
  for (int32_t i = 0; i < group_size / UNROLL; i++) {
    float4 rand = curand_uniform4(state);
    for (int32_t j = 0; j < UNROLL; j++) {
      int pos = i * UNROLL + j;
      int32_t r = (int32_t)((&rand.x)[j] * group_size);
      
      int32_t tmp = heap[pos];
      heap[pos] = heap[r];
      heap[r] = tmp;
    }
  }

  // assign nonzero elements to y
  for (int32_t i = 0; i < nonzero; i++) {
    y[heap[i]] = x[heap[i]];
  }
}

template <typename scalar_t>
__device__ void
prune_rnd_channelwise(const scalar_t *__restrict__ x, scalar_t *__restrict__ y,
                      curandStatePhilox4_32_10_t *state,
                      const int32_t nonzero, const int32_t group_size,
                      const int32_t B, const int32_t C, const int32_t W,
                      const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t b = get_0dim(idx, C, W, H);
  const int32_t c = get_1dim(idx, C, W, H);
  const int32_t w = get_2dim(idx, C, W, H);
  const int32_t h = get_3dim(idx, C, W, H);

  if (b >= B || c >= C / group_size || w >= W || h >= H) {
    return;
  }

  int32_t heap[MAX_GROUP_SIZE];

  // Randomly select nonzero elements
  for (int32_t i = 0; i < group_size; i++) {
    int32_t ci = c * group_size + i;
    // printf("idx(%d) %d\n", idx, ci);
    heap[i] = idx_4d(b, ci, w, h, C, W, H);
  }

  // Randomly shuffle the heap
  for (int32_t i = 0; i < group_size / UNROLL; i++) {
    float4 rand = curand_uniform4(state);
    for (int32_t j = 0; j < UNROLL; j++) {
      int pos = i * UNROLL + j;
      int32_t r = (int32_t)((&rand.x)[j] * group_size);
      
      int32_t tmp = heap[pos];
      heap[pos] = heap[r];
      heap[r] = tmp;
    }
  }

  // assign nonzero elements to y
  for (int32_t i = 0; i < nonzero; i++) {
    y[heap[i]] = x[heap[i]];
  }
}

template <typename scalar_t>
__global__ void
prune_kernel(const scalar_t *__restrict__ x, scalar_t *__restrict__ y,
             at::PhiloxCudaState philox_args,
             const int32_t prune_type, const int32_t prune_dim,
             const int32_t nonzero, const int32_t group_size, const int32_t B,
             const int32_t C, const int32_t W, const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t c = get_1dim(idx, C, W, H);

  // if (idx >= B * W * H * C) {
  //   return;
  // }
  auto seeds = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  // Select proper pruning algorithm and call it
  if (prune_type == PRUNE_MAX && prune_dim == 0) {
    prune_max_batchwise<scalar_t>(x, y, nonzero, group_size, B, C, W, H);
  } else if (prune_type == PRUNE_MAX && prune_dim == 1) {
    prune_max_channelwise<scalar_t>(x, y, nonzero, group_size, B, C, W, H);
  } else if (prune_type == PRUNE_RND && prune_dim == 0) {
    prune_rnd_batchwise<scalar_t>(x, y, &state, nonzero, group_size, B, C, W, H);
  } else if (prune_type == PRUNE_RND && prune_dim == 1) {
    prune_rnd_channelwise<scalar_t>(x, y, &state, nonzero, group_size, B, C, W, H);
  }
}
} // namespace


at::Tensor prune_cuda(const at::Tensor x, const int prune_type,
                      const int prune_dim, const int nonzero,
                      const int group_size) {
  const auto B = x.size(0);
  const auto C = x.size(1);
  const auto W = x.size(2);
  const auto H = x.size(3);
  const auto size = B * C * W * H;
  auto y = at::zeros_like(x);
  const int threads = 256;
  int blocks;
  if (prune_dim == 0) {
    blocks = ((B / group_size) * C * W * H) / threads + 1;
  } else if (prune_dim == 1) {
    blocks = ((C / group_size) * B * W * H) / threads + 1;
  } else {
    AT_ERROR("prune_dim must be 0 or 1");
  }

  auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator(x.get_device()));
  int32_t counter_offset = 1;
  at::PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  AT_DISPATCH_FLOATING_TYPES(x.type(), "prune_cuda", ([&] {
                               prune_kernel<scalar_t><<<blocks, threads>>>(
                                   x.data<scalar_t>(), y.data<scalar_t>(), 
                                   rng_engine_inputs,
                                   prune_type, prune_dim, nonzero, group_size,
                                   B, C, W, H);
                             }));
                         
  return y;
}
