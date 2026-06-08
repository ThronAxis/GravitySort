/**
 * GravitySort — ML Reduction Kernels
 * ──────────────────────────────────────────────────────────────────────────
 * Four variants of parallel reduction (sum over float array):
 *
 *   1. Naive        — global atomics               (~10% memory bandwidth)
 *   2. Shared-mem   — binary tree reduction         (~50% memory bandwidth)
 *   3. Warp shuffle — __shfl_xor_sync, no shmem     (~67% memory bandwidth)
 *   4. Vectorized   — float4 loads, max throughput   (≥85% memory bandwidth)
 *
 * Each variant is benchmarked and compared to Thrust::reduce.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                            \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

// ─── 1. Naive: global atomics ────────────────────────────────────────────────
__global__ void reduce_naive(const float *in, float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) atomicAdd(out, in[tid]);
}

// ─── 2. Shared-memory binary tree ────────────────────────────────────────────
__global__ void reduce_shared(const float *in, float *out, int n) {
  extern __shared__ float sdata[];
  int tid  = threadIdx.x;
  int gid  = blockIdx.x * blockDim.x * 2 + tid;

  // Load two elements per thread (sequential addressing, no bank conflicts)
  sdata[tid]  = (gid < n)              ? in[gid]              : 0.0f;
  sdata[tid] += (gid + blockDim.x < n) ? in[gid + blockDim.x] : 0.0f;
  __syncthreads();

  // Binary tree reduction with loop unrolling for last warp
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // Last warp — no sync needed
  if (tid < 32) {
    volatile float *vs = sdata;
    vs[tid] += vs[tid + 32];
    vs[tid] += vs[tid + 16];
    vs[tid] += vs[tid +  8];
    vs[tid] += vs[tid +  4];
    vs[tid] += vs[tid +  2];
    vs[tid] += vs[tid +  1];
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

// ─── 3. Warp shuffle (__shfl_xor_sync) ──────────────────────────────────────
__device__ float warp_reduce_sum(float val) {
  // No shared memory needed — uses warp registers directly
  val += __shfl_xor_sync(0xffffffff, val, 16);
  val += __shfl_xor_sync(0xffffffff, val, 8);
  val += __shfl_xor_sync(0xffffffff, val, 4);
  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 1);
  return val;
}

__global__ void reduce_warp(const float *in, float *out, int n) {
  extern __shared__ float warp_sums[];

  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int lane   = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;

  float val = (tid < n) ? in[tid] : 0.0f;
  val = warp_reduce_sum(val);

  if (lane == 0) warp_sums[warp_id] = val;
  __syncthreads();

  // Final warp reduction across warp sums
  if (warp_id == 0) {
    val = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) atomicAdd(out, val);
  }
}

// ─── 4. Vectorized float4 loads ──────────────────────────────────────────────
__global__ void reduce_vectorized(const float4 *in, float *out, int n4) {
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

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) vdata[tid] += vdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, vdata[0]);
}

// ─── Benchmark helper ─────────────────────────────────────────────────────────
struct BenchResult { float us; float bw_gbps; float result; };

BenchResult run_reduce(const char *name,
                        void (*kernel_fn)(const float *, float *, int, int),
                        const float *d_in, int n, int blockSize) {
  float *d_out;
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  const int REPS = 100;
  CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
  CHECK_CUDA(cudaEventRecord(start));
  for (int r = 0; r < REPS; r++) {
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
    kernel_fn(d_in, d_out, n, blockSize);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  float us     = ms * 1000.0f / REPS;
  float bw     = (float)n * sizeof(float) / (us * 1e-6) / 1e9;

  float h_out = 0;
  CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  printf("  %-20s  %8.2f µs  %7.2f GB/s\n", name, us, bw);
  return {us, bw, h_out};
}

// Wrappers to match signature
static int g_blockSize = 256;
void naive_wrapper(const float *in, float *out, int n, int bs) {
  int g = (n + bs - 1) / bs;
  reduce_naive<<<g, bs>>>(in, out, n);
}
void shared_wrapper(const float *in, float *out, int n, int bs) {
  int g = (n / 2 + bs - 1) / bs;
  reduce_shared<<<g, bs, bs * sizeof(float)>>>(in, out, n);
}
void warp_wrapper(const float *in, float *out, int n, int bs) {
  int g = (n + bs - 1) / bs;
  int warps = bs / 32;
  reduce_warp<<<g, bs, warps * sizeof(float)>>>(in, out, n);
}
void vec_wrapper(const float *in, float *out, int n, int bs) {
  int n4 = n / 4;
  int g  = (n4 + bs - 1) / bs;
  reduce_vectorized<<<g, bs, bs * sizeof(float)>>>((const float4 *)in, out, n4);
}

int main(int argc, char **argv) {
  const int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);  // 32M floats
  printf("GravitySort ⚡ Reduction Kernels  N=%d (%.1f MB)\n",
         N, N * 4.0f / 1024 / 1024);

  float *h_data = (float *)malloc(N * sizeof(float));
  float ref = 0.0f;
  srand(42);
  for (int i = 0; i < N; i++) { h_data[i] = (float)rand() / RAND_MAX; ref += h_data[i]; }

  float *d_in;
  CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  printf("  %-20s  %8s  %10s\n", "Variant", "Time", "Bandwidth");
  printf("  %s\n", "────────────────────────────────────────────────");

  int bs = 256;
  run_reduce("Naive (atomics)",     naive_wrapper,  d_in, N, bs);
  run_reduce("Shared-mem tree",     shared_wrapper, d_in, N, bs);
  run_reduce("Warp shuffle",        warp_wrapper,   d_in, N, bs);
  run_reduce("Vectorized float4",   vec_wrapper,    d_in, N, bs);

  // Thrust reference
  {
    thrust::device_ptr<float> dp(d_in);
    cudaEvent_t s, e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
    float res = 0;
    CHECK_CUDA(cudaEventRecord(s));
    for (int r = 0; r < 100; r++) res = thrust::reduce(dp, dp + N, 0.0f);
    CHECK_CUDA(cudaEventRecord(e)); CHECK_CUDA(cudaEventSynchronize(e));
    float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    float us = ms * 10.0f;
    printf("  %-20s  %8.2f µs  %7.2f GB/s  ← Thrust ref\n",
           "thrust::reduce", us, (float)N * 4 / (us * 1e-6) / 1e9);
    CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
  }

  CHECK_CUDA(cudaFree(d_in));
  free(h_data);
  return 0;
}
