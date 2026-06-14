/**
 * GravitySort — Reduction Kernels (Shared Header)
 * ─────────────────────────────────────────────────────────────────────────────
 * Defines all 4 reduction kernel variants for inclusion in both
 * the demo (reduction.cu) and benchmarks (bench_reduce.cu).
 *
 * All kernels compute: out += sum(in[0..n-1])  (accumulate via atomicAdd)
 *
 *  Variant 1 — Naive        : global atomics only
 *  Variant 2 — Shared-mem   : binary tree in shared memory
 *  Variant 3 — Warp shuffle : __shfl_xor_sync, no shared mem needed
 *  Variant 4 — Vectorized   : float4 loads for maximum memory throughput
 */

#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ─── 1. Naive: one atomic per thread ─────────────────────────────────────────
__global__ void kernel_reduce_naive(const float *__restrict__ in,
                                    float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) atomicAdd(out, in[tid]);
}

// ─── 2. Shared-memory binary tree ────────────────────────────────────────────
__global__ void kernel_reduce_shared(const float *__restrict__ in,
                                     float *out, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x * 2 + tid;

  sdata[tid]  = (gid < n)              ? in[gid]              : 0.0f;
  sdata[tid] += (gid + blockDim.x < n) ? in[gid + blockDim.x] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid < 32) {
    volatile float *vs = sdata;
    vs[tid] += vs[tid + 32]; vs[tid] += vs[tid + 16];
    vs[tid] += vs[tid +  8]; vs[tid] += vs[tid +  4];
    vs[tid] += vs[tid +  2]; vs[tid] += vs[tid +  1];
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

// ─── 3. Warp shuffle — __shfl_xor_sync, zero shared-mem overhead ─────────────
__device__ __forceinline__ float _gs_warp_sum(float val) {
  val += __shfl_xor_sync(0xffffffff, val, 16);
  val += __shfl_xor_sync(0xffffffff, val,  8);
  val += __shfl_xor_sync(0xffffffff, val,  4);
  val += __shfl_xor_sync(0xffffffff, val,  2);
  val += __shfl_xor_sync(0xffffffff, val,  1);
  return val;
}

__global__ void kernel_reduce_warp(const float *__restrict__ in,
                                   float *out, int n) {
  extern __shared__ float warp_sums[];
  int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  int lane    = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;

  float val = (tid < n) ? in[tid] : 0.0f;
  val = _gs_warp_sum(val);

  if (lane == 0) warp_sums[warp_id] = val;
  __syncthreads();

  if (warp_id == 0) {
    val = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    val = _gs_warp_sum(val);
    if (lane == 0) atomicAdd(out, val);
  }
}

// ─── 4. Vectorized float4 — maximises HBM throughput ────────────────────────
__global__ void kernel_reduce_vec4(const float4 *__restrict__ in,
                                   float *out, int n4) {
  extern __shared__ float vdata[];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  float sum = 0.0f;
  if (gid < n4) {
    float4 v = in[gid];
    sum = v.x + v.y + v.z + v.w;
  }
  vdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) vdata[tid] += vdata[tid + s];
    __syncthreads();
  }
  if (tid < 32) {
    volatile float *vs = vdata;
    vs[tid] += vs[tid + 32]; vs[tid] += vs[tid + 16];
    vs[tid] += vs[tid +  8]; vs[tid] += vs[tid +  4];
    vs[tid] += vs[tid +  2]; vs[tid] += vs[tid +  1];
  }
  if (tid == 0) atomicAdd(out, vdata[0]);
}

// ─── Launch helpers ───────────────────────────────────────────────────────────
inline void launch_reduce_naive(const float *d, float *out, int n, int bs=256) {
  int g = (n + bs - 1) / bs;
  kernel_reduce_naive<<<g, bs>>>(d, out, n);
}
inline void launch_reduce_shared(const float *d, float *out, int n, int bs=256) {
  int g = (n / 2 + bs - 1) / bs;
  kernel_reduce_shared<<<g, bs, bs * sizeof(float)>>>(d, out, n);
}
inline void launch_reduce_warp(const float *d, float *out, int n, int bs=256) {
  int g = (n + bs - 1) / bs;
  int warps = bs / 32;
  kernel_reduce_warp<<<g, bs, warps * sizeof(float)>>>(d, out, n);
}
inline void launch_reduce_vec4(const float *d, float *out, int n, int bs=256) {
  int n4 = n / 4;
  int g  = (n4 + bs - 1) / bs;
  kernel_reduce_vec4<<<g, bs, bs * sizeof(float)>>>((const float4 *)d, out, n4);
}
