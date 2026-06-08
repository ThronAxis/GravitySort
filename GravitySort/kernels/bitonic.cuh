/**
 * GravitySort — Bitonic Sort Public Header
 * ──────────────────────────────────────────────────────────────────────────
 * Declares the host-callable bitonic sort interface for int32, float32, uint64.
 * Implementation: kernels/bitonic.cu
 */

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * bitonic_sort_f32
 * Sort an array of float32 values in ascending order on the GPU.
 *
 * @param data   Host pointer to the input array (modified in place)
 * @param n      Number of elements (any size; padded to next power-of-2 internally)
 * @param stream CUDA stream to use (0 = default stream)
 * @return       Kernel elapsed time in microseconds (µs)
 */
float bitonic_sort_f32(float *data, int n, cudaStream_t stream = 0);

/**
 * bitonic_sort_i32
 * Sort an array of int32 values in ascending order on the GPU.
 */
float bitonic_sort_i32(int *data, int n, cudaStream_t stream = 0);

/**
 * bitonic_sort_u64
 * Sort an array of uint64 values in ascending order on the GPU.
 */
float bitonic_sort_u64(uint64_t *data, int n, cudaStream_t stream = 0);

/**
 * bitonic_sort_device
 * Sort a device-resident array (no H2D/D2H transfers).
 * Caller is responsible for allocating d_data and d_tmp (same size).
 *
 * @param d_data Device pointer (in/out)
 * @param n      Number of elements
 * @param stream CUDA stream
 * @return       Kernel time in µs
 */
float bitonic_sort_device(int *d_data, int n, cudaStream_t stream = 0);

/**
 * bitonic_print_occupancy
 * Prints CUDA occupancy stats for the bitonic kernels on the current device.
 */
void bitonic_print_occupancy();

#ifdef __cplusplus
}
#endif

// ─── C++ template interface (header-only convenience) ─────────────────────
#ifdef __cplusplus

#include <type_traits>

/**
 * Usage:
 *   std::vector<int> v = { ... };
 *   float us = bitonic_sort(v.data(), v.size());
 */
template <typename T>
float bitonic_sort(T *data, int n, cudaStream_t stream = 0) {
  static_assert(
    std::is_same<T, int>::value ||
    std::is_same<T, float>::value ||
    std::is_same<T, uint64_t>::value,
    "bitonic_sort: only int32, float32, uint64 supported"
  );
  if constexpr (std::is_same<T, float>::value)
    return bitonic_sort_f32(data, n, stream);
  else if constexpr (std::is_same<T, int>::value)
    return bitonic_sort_i32(data, n, stream);
  else
    return bitonic_sort_u64(data, n, stream);
}

#endif  // __cplusplus
