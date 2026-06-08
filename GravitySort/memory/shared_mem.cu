/**
 * GravitySort — Shared Memory Bank Conflict Analysis
 * ──────────────────────────────────────────────────────────────────────────
 * Demonstrates:
 *   • Bank conflicts on 32-bank shared memory (stride-32 access pattern)
 *   • Fix via +1 padding (stride = TILE+1) → zero conflicts
 *   • L1/shared hit rate profiling (measure via Nsight: ncu --metric ...)
 *   • All global reads/writes 128-byte aligned
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

#define TILE 32

// ─── Conflicted: strided shared memory access ─────────────────────────────
// Each warp accesses column-major → all 32 threads hit same bank
__global__ void transpose_conflicted(const float *in, float *out,
                                      int width, int height) {
  __shared__ float tile[TILE][TILE];   // ← no padding: bank conflicts!
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  __syncthreads();

  int tx = blockIdx.y * TILE + threadIdx.x;
  int ty = blockIdx.x * TILE + threadIdx.y;
  if (tx < height && ty < width) out[ty * height + tx] = tile[threadIdx.x][threadIdx.y];
}

// ─── Fixed: +1 stride padding → zero bank conflicts ──────────────────────
__global__ void transpose_no_conflict(const float *in, float *out,
                                       int width, int height) {
  __shared__ float tile[TILE][TILE + 1];  // ← +1 pad: all different banks
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  __syncthreads();

  int tx = blockIdx.y * TILE + threadIdx.x;
  int ty = blockIdx.x * TILE + threadIdx.y;
  if (tx < height && ty < width) out[ty * height + tx] = tile[threadIdx.x][threadIdx.y];
}

// ─── Benchmark helper ─────────────────────────────────────────────────────
float benchmark_kernel(void (*fn)(const float *, float *, int, int),
                        const float *d_in, float *d_out, int W, int H) {
  dim3 block(TILE, TILE);
  dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  const int REPS = 200;
  CHECK_CUDA(cudaEventRecord(start));
  for (int r = 0; r < REPS; r++) fn<<<grid, block>>>(d_in, d_out, W, H);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms * 1000.0f / REPS;
}

int main() {
  // Use power-of-2 dimensions for clean alignment
  const int W = 4096, H = 4096;
  size_t bytes = (size_t)W * H * sizeof(float);

  float *d_in, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in,  bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));

  // Initialize
  float *h_in = (float *)malloc(bytes);
  for (int i = 0; i < W * H; i++) h_in[i] = (float)i;
  CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  printf("GravitySort ⚡ Shared Memory Bank Conflict Analysis\n");
  printf("  Matrix: %d × %d (%.1f MB)\n", W, H, bytes / 1024.0 / 1024.0);
  printf("  %-25s  %8s  %10s\n", "Kernel", "Time µs", "BW GB/s");
  printf("  %s\n", "────────────────────────────────────────────────");

  float us_bad  = benchmark_kernel(transpose_conflicted,   d_in, d_out, W, H);
  float us_good = benchmark_kernel(transpose_no_conflict,  d_in, d_out, W, H);
  float bw_bad  = 2.0f * bytes / (us_bad  * 1e-6) / 1e9;
  float bw_good = 2.0f * bytes / (us_good * 1e-6) / 1e9;

  printf("  %-25s  %8.2f  %10.2f  ← bank conflicts\n",  "Conflicted (stride-32)", us_bad,  bw_bad);
  printf("  %-25s  %8.2f  %10.2f  ← +1 pad fix\n",       "No-conflict (+1 pad)",  us_good, bw_good);
  printf("  Speedup from fix: %.2fx\n", us_bad / us_good);
  printf("\n  Run: ncu --metric l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum \\\n");
  printf("           ./shared_mem_demo\n");
  printf("  to see bank conflict counts in Nsight Compute.\n");

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  free(h_in);
  return 0;
}
