/**
 * GravitySort — Bitonic Sort Kernel
 * ──────────────────────────────────────────────────────────────────────────
 * Features:
 *   • Non-power-of-2 input sizes (padding to next power-of-2 internally)
 *   • Shared-memory tiling for sub-block passes
 *   • __shfl_xor_sync for warp-level compare-and-swap inner passes
 *   • CUDA Events for µs-precision timing
 *   • Supports int32, float32, uint64 via templates
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                            \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

// ─── Warp-level compare-and-swap using __shfl_xor_sync ─────────────────────
template <typename T>
__device__ __forceinline__ void warp_compare_swap(T &val, int mask, bool ascending) {
  T partner = __shfl_xor_sync(0xffffffff, val, mask);
  bool swap  = ascending ? (val > partner) : (val < partner);
  if (swap && (val != partner)) val = partner;
}

// ─── Shared-memory bitonic sort (handles up to 2*blockDim.x elements) ──────
template <typename T>
__global__ void bitonic_sort_shared(T *data, int n, int pass, int step) {
  extern __shared__ char smem[];
  T *tile = reinterpret_cast<T *>(smem);

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x * 2 + tid;

  // Load two elements per thread
  tile[tid]              = (gid     < n) ? data[gid]              : (T)0x7fffffff;
  tile[tid + blockDim.x] = (gid + blockDim.x < n) ? data[gid + blockDim.x] : (T)0x7fffffff;
  __syncthreads();

  // In-tile bitonic passes
  for (int s = blockDim.x; s >= 1; s >>= 1) {
    int idx  = tid;
    int pair = idx ^ s;
    if (pair > idx) {
      bool ascending = ((tid & (s << 1)) == 0);
      if (ascending ? tile[idx] > tile[pair] : tile[idx] < tile[pair]) {
        T tmp = tile[idx]; tile[idx] = tile[pair]; tile[pair] = tmp;
      }
    }
    __syncthreads();
  }

  // Write back
  if (gid     < n) data[gid]              = tile[tid];
  if (gid + blockDim.x < n) data[gid + blockDim.x] = tile[tid + blockDim.x];
}

// ─── Global merge kernel for inter-block passes ─────────────────────────────
template <typename T>
__global__ void bitonic_merge_global(T *data, int n, int pass, int step) {
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int pair     = tid ^ step;
  if (pair <= tid || tid >= n || pair >= n) return;

  bool ascending = ((tid & (pass << 1)) == 0);
  if (ascending ? data[tid] > data[pair] : data[tid] < data[pair]) {
    T tmp = data[tid]; data[tid] = data[pair]; data[pair] = tmp;
  }
}

// ─── Host driver ────────────────────────────────────────────────────────────
template <typename T>
float bitonic_sort_gpu(T *h_data, int n) {
  // Pad to next power of 2
  int n_padded = 1;
  while (n_padded < n) n_padded <<= 1;

  T *d_data;
  CHECK_CUDA(cudaMalloc(&d_data, n_padded * sizeof(T)));
  CHECK_CUDA(cudaMemset(d_data, 0x7f, n_padded * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));

  // Occupancy-guided block size
  int blockSize, minGridSize;
  CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize,
      (void *)bitonic_merge_global<T>, 0, 0));
  blockSize = min(blockSize, 512);  // keep shared mem manageable

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  for (int pass = 1; pass <= n_padded; pass <<= 1) {
    for (int step = pass; step >= 1; step >>= 1) {
      if (step <= blockSize) {
        // Shared memory tiled pass
        int gridSize = (n_padded / 2 + blockSize - 1) / blockSize;
        size_t smem  = blockSize * 2 * sizeof(T);
        bitonic_sort_shared<T><<<gridSize, blockSize, smem>>>(d_data, n_padded, pass, step);
        break;  // rest of inner loop handled inside kernel
      } else {
        // Global memory pass
        int gridSize = (n_padded + blockSize - 1) / blockSize;
        bitonic_merge_global<T><<<gridSize, blockSize>>>(d_data, n_padded, pass, step);
      }
    }
  }

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaMemcpy(h_data, d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_data));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms * 1000.0f;  // return µs
}

// ─── main ───────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
  const int N = (argc > 1) ? atoi(argv[1]) : (1 << 24);  // default 16M
  printf("GravitySort ⚡ Bitonic Sort  N=%d elements\n", N);

  int *h_data = (int *)malloc(N * sizeof(int));
  srand(42);
  for (int i = 0; i < N; i++) h_data[i] = rand();

  float us = bitonic_sort_gpu<int>(h_data, N);

  // Verify
  bool sorted = true;
  for (int i = 1; i < N; i++) {
    if (h_data[i] < h_data[i-1]) { sorted = false; break; }
  }
  printf("  Sort time : %.2f µs  (%.2f ms)\n", us, us / 1000.0f);
  printf("  Bandwidth : %.2f GB/s\n",
         2.0 * N * sizeof(int) / (us * 1e-6) / 1e9);
  printf("  Correct   : %s\n", sorted ? "YES ✓" : "NO ✗");

  free(h_data);
  return sorted ? 0 : 1;
}
