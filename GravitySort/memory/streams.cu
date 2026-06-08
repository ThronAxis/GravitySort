/**
 * GravitySort — Stream Concurrency Demo
 * ──────────────────────────────────────────────────────────────────────────
 * Demonstrates overlapping:
 *   Stream 0: H2D transfer   (chunk 0)
 *   Stream 1: Kernel execute (chunk 0, after its H2D)
 *   Stream 2: D2H transfer   (chunk 0, after its kernel)
 *
 * Uses 3 independent CUDA streams and pinned (page-locked) host memory
 * so that H2D/D2H truly overlaps kernel execution on the SM.
 *
 * Visualise the overlap with:
 *   nsys profile --trace=cuda ./streams_demo
 *   nsys-ui report1.nsys-rep
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                            \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

#define NUM_STREAMS 3
#define CHUNK_SIZE  (1 << 22)   // 4M floats per chunk = 16 MB

// ─── A simple compute kernel (scale array by constant) ────────────────────
__global__ void scale_kernel(float *data, int n, float alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // No data-dependent branching in inner loop (PRD requirement)
  if (tid < n) data[tid] *= alpha;
}

int main() {
  const int N_CHUNKS = 6;            // process 6 chunks in pipeline
  const int N_TOTAL  = N_CHUNKS * CHUNK_SIZE;

  printf("GravitySort ⚡ Stream Concurrency Demo\n");
  printf("  %d chunks × %d floats = %.1f MB total\n",
         N_CHUNKS, CHUNK_SIZE, N_TOTAL * 4.0f / 1024 / 1024);

  // ── Pinned host memory (required for async transfers) ───────────────────
  float *h_in, *h_out;
  CHECK_CUDA(cudaMallocHost(&h_in,  N_TOTAL * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_out, N_TOTAL * sizeof(float)));
  for (int i = 0; i < N_TOTAL; i++) h_in[i] = (float)(i % 1000);

  // ── Device buffers (one pair per stream) ────────────────────────────────
  float *d_buf[NUM_STREAMS];
  for (int s = 0; s < NUM_STREAMS; s++)
    CHECK_CUDA(cudaMalloc(&d_buf[s], CHUNK_SIZE * sizeof(float)));

  // ── Create streams ───────────────────────────────────────────────────────
  cudaStream_t streams[NUM_STREAMS];
  for (int s = 0; s < NUM_STREAMS; s++)
    CHECK_CUDA(cudaStreamCreate(&streams[s]));

  int blockSize = 256;
  int gridSize  = (CHUNK_SIZE + blockSize - 1) / blockSize;
  const float ALPHA = 2.0f;

  cudaEvent_t ev_start, ev_stop;
  CHECK_CUDA(cudaEventCreate(&ev_start));
  CHECK_CUDA(cudaEventCreate(&ev_stop));
  CHECK_CUDA(cudaEventRecord(ev_start, streams[0]));

  // ── Pipelined execution ──────────────────────────────────────────────────
  // Dispatch all chunks; GPU overlaps H2D → kernel → D2H across streams
  for (int c = 0; c < N_CHUNKS; c++) {
    int s       = c % NUM_STREAMS;                    // round-robin stream
    float *h_in_chunk  = h_in  + (size_t)c * CHUNK_SIZE;
    float *h_out_chunk = h_out + (size_t)c * CHUNK_SIZE;

    // H2D
    CHECK_CUDA(cudaMemcpyAsync(d_buf[s], h_in_chunk,
                               CHUNK_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice, streams[s]));
    // Kernel
    scale_kernel<<<gridSize, blockSize, 0, streams[s]>>>(d_buf[s], CHUNK_SIZE, ALPHA);

    // D2H
    CHECK_CUDA(cudaMemcpyAsync(h_out_chunk, d_buf[s],
                               CHUNK_SIZE * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[s]));
  }

  CHECK_CUDA(cudaEventRecord(ev_stop, streams[(N_CHUNKS - 1) % NUM_STREAMS]));

  // Sync all streams
  for (int s = 0; s < NUM_STREAMS; s++)
    CHECK_CUDA(cudaStreamSynchronize(streams[s]));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));

  // Verify first few elements
  bool ok = true;
  for (int i = 0; i < 1000 && ok; i++)
    if (fabsf(h_out[i] - h_in[i] * ALPHA) > 1e-4f) ok = false;

  printf("  Total time  : %.2f ms\n", ms);
  printf("  Throughput  : %.2f GB/s (H2D+kernel+D2H)\n",
         2.0f * N_TOTAL * 4 / (ms * 1e-3) / 1e9);
  printf("  Correct     : %s\n", ok ? "YES ✓" : "NO ✗");
  printf("\n  Visualise overlap:\n");
  printf("    nsys profile --trace=cuda ./streams_demo\n");
  printf("    nsys-ui report1.nsys-rep\n");

  // Cleanup
  for (int s = 0; s < NUM_STREAMS; s++) {
    CHECK_CUDA(cudaFree(d_buf[s]));
    CHECK_CUDA(cudaStreamDestroy(streams[s]));
  }
  CHECK_CUDA(cudaFreeHost(h_in));
  CHECK_CUDA(cudaFreeHost(h_out));
  CHECK_CUDA(cudaEventDestroy(ev_start));
  CHECK_CUDA(cudaEventDestroy(ev_stop));
  return ok ? 0 : 1;
}
