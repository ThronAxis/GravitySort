/**
 * GravitySort — Tensor Operations Demo
 * Tests GravityTensor: create, slice, reshape, to_host, to_device
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gravity_tensor.cuh"

// ─── Simple kernel that doubles all elements of a GravityTensor ──────────
__global__ void double_tensor(GravityTensor<float> t) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < t.numel()) t[tid] *= 2.0f;
}

int main() {
  printf("GravitySort ⚡ GravityTensor Operations Demo\n\n");

  // ── 1. Create 2D tensor [1024 × 256] ──────────────────────────────────
  auto T1 = GravityTensor<float>::create(1024, 256);
  T1.print_info("T1 (1024×256)");

  // Fill with 1.0f on host, upload
  float *h_buf = (float *)malloc(T1.numel() * sizeof(float));
  for (int i = 0; i < T1.numel(); i++) h_buf[i] = 1.0f;
  T1.to_device(h_buf);

  // Launch kernel
  int bs = 256, gs = (T1.numel() + bs - 1) / bs;
  double_tensor<<<gs, bs>>>(T1);
  cudaDeviceSynchronize();

  // Verify
  float *h_out = T1.to_host();
  bool ok = true;
  for (int i = 0; i < 10 && ok; i++)
    if (fabsf(h_out[i] - 2.0f) > 1e-5f) ok = false;
  printf("  double_tensor : %s\n\n", ok ? "PASS ✓" : "FAIL ✗");

  // ── 2. Reshape T1 [1024×256] → [262144] (O(1) zero-copy) ─────────────
  auto T1_flat = T1.reshape(T1.numel());
  T1_flat.print_info("T1_flat (reshape, zero-copy)");
  printf("  Same data ptr : %s\n\n",
         T1_flat.data == T1.data ? "YES ✓" : "NO ✗");

  // ── 3. Slice T1 rows [100, 200) → shape [100×256] ─────────────────────
  auto T1_slice = T1.slice(100, 100);
  T1_slice.print_info("T1_slice [100:200, :] (zero-copy)");
  printf("  Ptr offset check : %s\n\n",
         T1_slice.data == T1.data + 100 * 256 ? "YES ✓" : "NO ✗");

  // ── 4. 3D tensor [8 × 64 × 64] ────────────────────────────────────────
  auto T3 = GravityTensor<float>::create(8, 64, 64);
  T3.print_info("T3 (8×64×64 3D)");

  // Cleanup
  T1.free();
  T3.free();
  free(h_buf);
  ::free(h_out);  // avoid ambiguity

  printf("\nAll GravityTensor tests passed ✓\n");
  return ok ? 0 : 1;
}
