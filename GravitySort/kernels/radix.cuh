/**
 * GravitySort — Radix Sort Public Header
 * ──────────────────────────────────────────────────────────────────────────
 * Declares the host-callable LSD Radix Sort interface.
 * Implementation: kernels/radix.cu
 *
 * Algorithm: 4-pass LSD (8 bits/pass) on 32-bit unsigned keys.
 * Pipeline:  [Histogram] → [Prefix Scan] → [Scatter]   (per pass)
 * Streaming: pass N histogram overlaps pass N-1 scatter on a 2nd stream.
 */

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Constants ───────────────────────────────────────────────────────────────
#define RADIX_BITS_PER_PASS   8
#define RADIX_BUCKETS         256     // 2^8
#define RADIX_PASSES          4       // 4 × 8 = 32-bit keys

/**
 * radix_sort_u32
 * In-place LSD Radix sort of uint32 array on GPU.
 *
 * @param data  Host pointer (modified in place)
 * @param n     Number of elements (any size)
 * @return      Total GPU kernel time in µs (excludes H2D/D2H)
 */
float radix_sort_u32(uint32_t *data, int n);

/**
 * radix_sort_u32_async
 * Async version using caller-supplied streams.
 * Overlaps H2D transfer on s_h2d with previous computation.
 *
 * @param data     Host pointer (pinned memory strongly recommended)
 * @param n        Number of elements
 * @param s_h2d    Stream for host-to-device transfer
 * @param s_kernel Stream for kernel execution
 * @param s_d2h    Stream for device-to-host transfer
 * @return         Kernel-only time in µs
 */
float radix_sort_u32_async(uint32_t *data, int n,
                            cudaStream_t s_h2d,
                            cudaStream_t s_kernel,
                            cudaStream_t s_d2h);

/**
 * radix_sort_device
 * Sort device-resident buffer — no host transfers.
 *
 * @param d_in   Input device buffer (will be modified; ping-pong internally)
 * @param d_tmp  Temp device buffer (same size as d_in) — caller allocates
 * @param n      Number of elements
 * @param stream CUDA stream
 * @return       Kernel time in µs
 */
float radix_sort_device(uint32_t *d_in, uint32_t *d_tmp,
                         int n, cudaStream_t stream = 0);

/**
 * radix_histogram
 * Compute per-digit frequency histogram (single pass exposed for testing).
 *
 * @param d_in      Input device array
 * @param d_hist    Output histogram [RADIX_BUCKETS] on device
 * @param n         Number of elements
 * @param bit_shift Which 8-bit digit (0, 8, 16, 24)
 * @param stream    CUDA stream
 */
void radix_histogram(const uint32_t *d_in, uint32_t *d_hist,
                     int n, int bit_shift, cudaStream_t stream = 0);

/**
 * radix_print_pipeline_stats
 * Prints per-pass timing breakdown (H2D / histogram / scan / scatter / D2H).
 */
void radix_print_pipeline_stats(uint32_t *data, int n);

#ifdef __cplusplus
}
#endif

// ─── C++ template convenience ─────────────────────────────────────────────
#ifdef __cplusplus

#include <vector>

/**
 * Sort a std::vector<uint32_t> in place on the GPU.
 * Returns elapsed kernel time in µs.
 *
 * Example:
 *   std::vector<uint32_t> v(1 << 24);
 *   std::iota(v.rbegin(), v.rend(), 0);
 *   float us = radix_sort(v);
 */
inline float radix_sort(std::vector<uint32_t> &v) {
  return radix_sort_u32(v.data(), static_cast<int>(v.size()));
}

#endif  // __cplusplus
