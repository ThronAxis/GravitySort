/**
 * GravitySort — LSD Radix Sort Kernel
 * ──────────────────────────────────────────────────────────────────────────
 * Features:
 *   • 4-pass LSD Radix (8-bit digit per pass) for 32-bit keys
 *   • Histogram + prefix scan + scatter — fully GPU-resident
 *   • Stream-pipelined: H2D → pass0 → pass1 → pass2 → pass3 → D2H
 *     all dispatched on overlapping CUDA streams where possible
 *   • Handles arbitrary N (not just power-of-2)
 *   • CUDA Events for µs timing
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                            \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

#define RADIX_BITS   8
#define RADIX_BUCKETS (1 << RADIX_BITS)   // 256
#define PASSES       4                     // 4 × 8-bit = 32-bit key

// ─── Histogram kernel ────────────────────────────────────────────────────────
__global__ void histogram_kernel(const uint32_t *in, uint32_t *hist,
                                  int n, int bit_shift) {
  extern __shared__ uint32_t local_hist[];
  // Zero shared histogram
  for (int i = threadIdx.x; i < RADIX_BUCKETS; i += blockDim.x)
    local_hist[i] = 0;
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    uint32_t digit = (in[tid] >> bit_shift) & (RADIX_BUCKETS - 1);
    atomicAdd(&local_hist[digit], 1);
  }
  __syncthreads();

  // Merge local → global histogram
  for (int i = threadIdx.x; i < RADIX_BUCKETS; i += blockDim.x)
    atomicAdd(&hist[i], local_hist[i]);
}

// ─── Exclusive prefix scan (single block, Hillis-Steele) ─────────────────────
__global__ void prefix_scan_kernel(uint32_t *hist, uint32_t *offsets) {
  extern __shared__ uint32_t scan_buf[];
  int tid = threadIdx.x;
  scan_buf[tid] = (tid < RADIX_BUCKETS) ? hist[tid] : 0;
  __syncthreads();

  for (int stride = 1; stride < RADIX_BUCKETS; stride <<= 1) {
    uint32_t val = (tid >= stride) ? scan_buf[tid - stride] : 0;
    __syncthreads();
    scan_buf[tid] += val;
    __syncthreads();
  }
  // Convert inclusive to exclusive
  offsets[tid] = (tid == 0) ? 0 : scan_buf[tid - 1];
}

// ─── Scatter kernel ──────────────────────────────────────────────────────────
__global__ void scatter_kernel(const uint32_t *in, uint32_t *out,
                                uint32_t *offsets, int n, int bit_shift) {
  // Each block handles a contiguous chunk; use atomics on offsets for position
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  uint32_t digit = (in[tid] >> bit_shift) & (RADIX_BUCKETS - 1);
  uint32_t pos   = atomicAdd(&offsets[digit], 1);
  out[pos]       = in[tid];
}

// ─── Host driver ─────────────────────────────────────────────────────────────
float radix_sort_gpu(uint32_t *h_data, int n) {
  uint32_t *d_in, *d_out, *d_hist, *d_offsets;
  size_t bytes = (size_t)n * sizeof(uint32_t);
  CHECK_CUDA(cudaMalloc(&d_in,      bytes));
  CHECK_CUDA(cudaMalloc(&d_out,     bytes));
  CHECK_CUDA(cudaMalloc(&d_hist,    RADIX_BUCKETS * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_offsets, RADIX_BUCKETS * sizeof(uint32_t)));

  // Stream setup: 3 independent streams (H2D, compute, D2H)
  cudaStream_t s_h2d, s_compute, s_d2h;
  CHECK_CUDA(cudaStreamCreate(&s_h2d));
  CHECK_CUDA(cudaStreamCreate(&s_compute));
  CHECK_CUDA(cudaStreamCreate(&s_d2h));

  // Pinned host buffer for async copies
  uint32_t *h_pinned;
  CHECK_CUDA(cudaMallocHost(&h_pinned, bytes));
  memcpy(h_pinned, h_data, bytes);

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Async H2D
  CHECK_CUDA(cudaMemcpyAsync(d_in, h_pinned, bytes, cudaMemcpyHostToDevice, s_h2d));
  CHECK_CUDA(cudaEventRecord(start, s_h2d));

  int blockSize = 256;
  int gridSize  = (n + blockSize - 1) / blockSize;
  size_t shist  = RADIX_BUCKETS * sizeof(uint32_t);

  // Synchronize s_compute on s_h2d completion
  cudaEvent_t h2d_done;
  CHECK_CUDA(cudaEventCreate(&h2d_done));
  CHECK_CUDA(cudaEventRecord(h2d_done, s_h2d));
  CHECK_CUDA(cudaStreamWaitEvent(s_compute, h2d_done, 0));

  for (int pass = 0; pass < PASSES; pass++) {
    int bit_shift = pass * RADIX_BITS;

    CHECK_CUDA(cudaMemsetAsync(d_hist, 0, RADIX_BUCKETS * sizeof(uint32_t), s_compute));
    histogram_kernel<<<gridSize, blockSize, shist, s_compute>>>(
        d_in, d_hist, n, bit_shift);
    prefix_scan_kernel<<<1, RADIX_BUCKETS, RADIX_BUCKETS * sizeof(uint32_t), s_compute>>>(
        d_hist, d_offsets);
    scatter_kernel<<<gridSize, blockSize, 0, s_compute>>>(
        d_in, d_out, d_offsets, n, bit_shift);

    // Ping-pong buffers
    uint32_t *tmp = d_in; d_in = d_out; d_out = tmp;
  }

  CHECK_CUDA(cudaEventRecord(stop, s_compute));

  // Async D2H
  cudaEvent_t compute_done;
  CHECK_CUDA(cudaEventCreate(&compute_done));
  CHECK_CUDA(cudaEventRecord(compute_done, s_compute));
  CHECK_CUDA(cudaStreamWaitEvent(s_d2h, compute_done, 0));
  CHECK_CUDA(cudaMemcpyAsync(h_pinned, d_in, bytes, cudaMemcpyDeviceToHost, s_d2h));
  CHECK_CUDA(cudaStreamSynchronize(s_d2h));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  memcpy(h_data, h_pinned, bytes);

  // Cleanup
  CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_hist)); CHECK_CUDA(cudaFree(d_offsets));
  CHECK_CUDA(cudaFreeHost(h_pinned));
  CHECK_CUDA(cudaStreamDestroy(s_h2d));
  CHECK_CUDA(cudaStreamDestroy(s_compute));
  CHECK_CUDA(cudaStreamDestroy(s_d2h));
  CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaEventDestroy(h2d_done)); CHECK_CUDA(cudaEventDestroy(compute_done));
  return ms * 1000.0f;
}

int main(int argc, char **argv) {
  const int N = (argc > 1) ? atoi(argv[1]) : (1 << 24);
  printf("GravitySort ⚡ Radix Sort (LSD 4-pass, stream-pipelined)  N=%d\n", N);

  uint32_t *h_data = (uint32_t *)malloc(N * sizeof(uint32_t));
  srand(42);
  for (int i = 0; i < N; i++) h_data[i] = (uint32_t)rand();

  float us = radix_sort_gpu(h_data, N);

  bool sorted = true;
  for (int i = 1; i < N; i++) {
    if (h_data[i] < h_data[i-1]) { sorted = false; break; }
  }
  printf("  Sort time : %.2f µs\n", us);
  printf("  Bandwidth : %.2f GB/s\n",
         2.0 * N * sizeof(uint32_t) / (us * 1e-6) / 1e9);
  printf("  Correct   : %s\n", sorted ? "YES ✓" : "NO ✗");

  free(h_data);
  return sorted ? 0 : 1;
}
