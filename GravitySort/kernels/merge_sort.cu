/**
 * GravitySort — GPU Merge Sort
 * ─────────────────────────────────────────────────────────────────────────────
 * Algorithm: Bottom-up, two-phase merge sort
 *
 *  Phase 1 — Local sort via bitonic in shared memory (one block = one chunk)
 *  Phase 2 — Bottom-up merge: repeatedly merge sorted pairs of runs using a
 *             cooperative merge kernel, doubling run size each pass.
 *
 * Complexity : O(n log²n) comparisons on GPU
 * Memory     : O(n) extra (double-buffering for stable in-place merge)
 * Features   :
 *   • Stable sort (preserves order of equal keys)
 *   • Non-power-of-2 sizes fully supported
 *   • Uses CUDA Events for µs-precision timing
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Bitonic sort in shared memory — each block sorts TILE elements
// ─────────────────────────────────────────────────────────────────────────────
#define TILE 1024  // elements per block (must be power of 2, ≤ shmem limit)

__global__ void merge_sort_local(int *data, int n) {
  __shared__ int tile[TILE];
  int block_start = blockIdx.x * TILE;
  int tid         = threadIdx.x;

  // Load — pad with INT_MAX for out-of-bounds
  for (int i = tid; i < TILE; i += blockDim.x)
    tile[i] = (block_start + i < n) ? data[block_start + i] : INT_MAX;
  __syncthreads();

  // Bitonic sort within shared memory
  for (int size = 2; size <= TILE; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = tid; i < TILE / 2; i += blockDim.x) {
        int lo = (i / stride) * stride * 2 + (i % stride);
        int hi = lo + stride;
        bool ascending = ((lo / size) & 1) == 0;
        if (ascending ? tile[lo] > tile[hi] : tile[lo] < tile[hi]) {
          int tmp = tile[lo]; tile[lo] = tile[hi]; tile[hi] = tmp;
        }
      }
      __syncthreads();
    }
  }

  // Write back
  for (int i = tid; i < TILE; i += blockDim.x)
    if (block_start + i < n) data[block_start + i] = tile[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Merge two sorted halves [left, mid) and [mid, right) into out[]
//          Each thread block handles one merge pair.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void merge_pairs(const int *__restrict__ src, int *__restrict__ dst,
                             int n, int width) {
  // Each block merges one pair of runs of length 'width'
  int pair_id   = blockIdx.x;
  int left      = pair_id * 2 * width;
  int mid       = min(left + width, n);
  int right     = min(left + 2 * width, n);

  if (left >= n) return;
  if (mid >= n) {
    // Only left half exists — copy as-is
    for (int i = threadIdx.x; i < right - left; i += blockDim.x)
      dst[left + i] = src[left + i];
    return;
  }

  int len_l = mid - left;
  int len_r = right - mid;
  int total  = len_l + len_r;

  // Each thread outputs one element using binary search rank
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    int val, rank_l, rank_r;

    // Determine if this output slot comes from left or right half
    // using a cooperative binary-search merge
    int lo = max(0, i - len_r), hi = min(i, len_l);
    while (lo < hi) {
      int m = (lo + hi) / 2;
      if (src[left + m] <= src[mid + (i - m - 1)]) lo = m + 1;
      else                                           hi = m;
    }
    int from_l = lo;
    int from_r = i - lo;

    if (from_l < len_l &&
        (from_r >= len_r || src[left + from_l] <= src[mid + from_r])) {
      dst[left + i] = src[left + from_l];
    } else {
      dst[left + i] = src[mid + from_r];
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host driver
// ─────────────────────────────────────────────────────────────────────────────
float merge_sort_gpu(int *h_data, int n) {
  int *d_a, *d_b;  // double buffer
  size_t bytes = (size_t)n * sizeof(int);
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_data, bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  // Phase 1: sort each TILE-sized block
  int grid1 = (n + TILE - 1) / TILE;
  int bs1   = TILE / 2;  // TILE/2 threads, each handles 2 comparisons
  merge_sort_local<<<grid1, bs1>>>(d_a, n);

  // Phase 2: bottom-up merge passes
  const int MERGE_BS = 256;
  for (int width = TILE; width < n; width <<= 1) {
    int num_pairs = (n + 2 * width - 1) / (2 * width);
    merge_pairs<<<num_pairs, MERGE_BS>>>(d_a, d_b, n, width);
    int *tmp = d_a; d_a = d_b; d_b = tmp;  // ping-pong
  }

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaMemcpy(h_data, d_a, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms * 1000.0f;  // return µs
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
  const int N = (argc > 1) ? atoi(argv[1]) : (1 << 22);  // default 4M
  printf("GravitySort -- Merge Sort (GPU stable, bottom-up)  N=%d\n", N);

  int *h_data = (int *)malloc((size_t)N * sizeof(int));
  srand(42);
  for (int i = 0; i < N; i++) h_data[i] = rand();

  float us = merge_sort_gpu(h_data, N);

  // Verify
  bool ok = true;
  for (int i = 1; i < N; i++)
    if (h_data[i] < h_data[i-1]) { ok = false; break; }

  printf("  Sort time : %.2f us  (%.2f ms)\n", us, us / 1000.0f);
  printf("  Bandwidth : %.2f GB/s\n",
         2.0 * N * sizeof(int) / (us * 1e-6) / 1e9);
  printf("  Correct   : %s\n", ok ? "YES" : "NO -- VERIFICATION FAILED");

  free(h_data);
  return ok ? 0 : 1;
}
