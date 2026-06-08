/**
 * GravitySort — Odd-Even (Brick) Sort Kernel
 * ──────────────────────────────────────────────────────────────────────────
 * Features:
 *   • Fully coalesced 128-byte aligned global memory accesses
 *   • Baseline reference implementation (O(n²) worst case)
 *   • Useful for demonstrating coalescing vs non-coalesced comparison
 *   • Supports int32, float32, uint64 via templates
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                            \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

// ─── One phase of odd-even sort (phase=0: even pairs, phase=1: odd pairs) ───
template <typename T>
__global__ void odd_even_phase(T *data, int n, int phase) {
  // Each thread handles one compare-and-swap pair
  // Coalesced: consecutive threads access consecutive memory addresses
  int tid   = blockIdx.x * blockDim.x + threadIdx.x;
  int start = tid * 2 + phase;         // phase selects even/odd pair offset

  if (start + 1 < n) {
    T a = data[start];
    T b = data[start + 1];
    if (a > b) {
      data[start]     = b;
      data[start + 1] = a;
    }
  }
}

// ─── Host driver ─────────────────────────────────────────────────────────────
template <typename T>
float odd_even_sort_gpu(T *h_data, int n) {
  T *d_data;
  CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));

  int blockSize, minGridSize;
  CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize, (void *)odd_even_phase<T>, 0, 0));
  int gridSize = ((n / 2) + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  for (int iter = 0; iter < n; iter++) {
    odd_even_phase<T><<<gridSize, blockSize>>>(d_data, n, iter % 2);
  }

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaMemcpy(h_data, d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_data));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms * 1000.0f;
}

int main(int argc, char **argv) {
  // Odd-even is O(n²) — keep N small for demo
  const int N = (argc > 1) ? atoi(argv[1]) : (1 << 16);  // default 64K
  printf("GravitySort ⚡ Odd-Even Sort (coalesced baseline)  N=%d\n", N);

  int *h_data = (int *)malloc(N * sizeof(int));
  srand(42);
  for (int i = 0; i < N; i++) h_data[i] = rand();

  float us = odd_even_sort_gpu<int>(h_data, N);

  bool sorted = true;
  for (int i = 1; i < N; i++) {
    if (h_data[i] < h_data[i-1]) { sorted = false; break; }
  }
  printf("  Sort time : %.2f µs\n", us);
  printf("  Correct   : %s\n", sorted ? "YES ✓" : "NO ✗");
  printf("  Note: O(n²) reference — use Bitonic/Radix for large N\n");

  free(h_data);
  return sorted ? 0 : 1;
}
