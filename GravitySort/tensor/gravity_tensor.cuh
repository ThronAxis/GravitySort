/**
 * GravitySort — GravityTensor<T> (GPU-Resident Tensor)
 * ──────────────────────────────────────────────────────────────────────────
 * GPU-resident struct supporting:
 *   • 1D / 2D / 3D tensors with row-major (C) stride ordering
 *   • slice()   — O(1) zero-copy view (adjust pointer + shape)
 *   • reshape() — O(1) zero-copy (update shape/strides only)
 *   • to_host() / to_device() — explicit data movement
 *   • Shape & stride metadata accessible from device code
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <typeinfo>

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                            \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

// ─── Max dimensionality ────────────────────────────────────────────────────
#define GRAVITY_MAX_DIMS 3

template <typename T>
struct GravityTensor {
  T       *data;                          // raw device pointer (owned if owns_data)
  int      shape[GRAVITY_MAX_DIMS];       // [d0, d1, d2] — unused dims = 1
  int      stride[GRAVITY_MAX_DIMS];      // row-major strides in elements
  int      ndim;                          // actual number of dimensions (1/2/3)
  bool     owns_data;                     // true iff this tensor allocated data

  // ── Constructors ──────────────────────────────────────────────────────────

  // Allocate new GPU tensor
  static GravityTensor<T> create(int d0, int d1 = 1, int d2 = 1) {
    GravityTensor<T> t;
    t.ndim = (d2 > 1) ? 3 : (d1 > 1) ? 2 : 1;
    t.shape[0]  = d0; t.shape[1]  = d1; t.shape[2]  = d2;
    // Row-major strides
    t.stride[2] = 1;
    t.stride[1] = d2;
    t.stride[0] = d1 * d2;
    t.owns_data = true;
    size_t bytes = (size_t)d0 * d1 * d2 * sizeof(T);
    CHECK_CUDA(cudaMalloc(&t.data, bytes));
    CHECK_CUDA(cudaMemset(t.data, 0, bytes));
    return t;
  }

  // View over existing device pointer (zero-copy)
  static GravityTensor<T> view(T *ptr, int d0, int d1 = 1, int d2 = 1) {
    GravityTensor<T> t;
    t.ndim = (d2 > 1) ? 3 : (d1 > 1) ? 2 : 1;
    t.shape[0]  = d0; t.shape[1]  = d1; t.shape[2]  = d2;
    t.stride[2] = 1;
    t.stride[1] = d2;
    t.stride[0] = d1 * d2;
    t.data      = ptr;
    t.owns_data = false;
    return t;
  }

  void free() {
    if (owns_data && data) { CHECK_CUDA(cudaFree(data)); data = nullptr; }
  }

  // ── numel ──────────────────────────────────────────────────────────────────
  __host__ __device__ int numel() const {
    return shape[0] * shape[1] * shape[2];
  }

  // ── reshape — O(1) zero-copy ───────────────────────────────────────────────
  // Returns a new view with the same data but different shape.
  // Requires total elements to match.
  GravityTensor<T> reshape(int d0, int d1 = 1, int d2 = 1) const {
    assert(d0 * d1 * d2 == numel());
    GravityTensor<T> t = view(data, d0, d1, d2);
    t.owns_data = false;
    return t;
  }

  // ── slice — O(1) zero-copy ─────────────────────────────────────────────────
  // Slice along dim 0: returns rows [start, start+len)
  GravityTensor<T> slice(int start, int len) const {
    assert(start + len <= shape[0]);
    GravityTensor<T> t;
    t.ndim = ndim;
    t.shape[0]  = len;
    t.shape[1]  = shape[1];
    t.shape[2]  = shape[2];
    t.stride[0] = stride[0];
    t.stride[1] = stride[1];
    t.stride[2] = stride[2];
    t.data      = data + (size_t)start * stride[0];
    t.owns_data = false;
    return t;
  }

  // ── to_host — copies device → host ────────────────────────────────────────
  T *to_host() const {
    size_t bytes = (size_t)numel() * sizeof(T);
    T *h = (T *)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(h, data, bytes, cudaMemcpyDeviceToHost));
    return h;   // caller must free()
  }

  // ── to_device — copies host → existing device tensor ─────────────────────
  void to_device(const T *h_src) {
    CHECK_CUDA(cudaMemcpy(data, h_src, (size_t)numel() * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  // ── Device-side element access (linear index) ─────────────────────────────
  __device__ __forceinline__ T &operator[](int i) { return data[i]; }
  __device__ __forceinline__ T  operator[](int i) const { return data[i]; }

  // ── 2D device access ──────────────────────────────────────────────────────
  __device__ __forceinline__ T &at(int i, int j) {
    return data[i * stride[0] + j * stride[1]];
  }

  // ── Print metadata ─────────────────────────────────────────────────────────
  void print_info(const char *name) const {
    printf("GravityTensor<%s> %s\n", typeid(T).name(), name);
    printf("  shape   : [%d, %d, %d]   ndim=%d\n",
           shape[0], shape[1], shape[2], ndim);
    printf("  strides : [%d, %d, %d]\n", stride[0], stride[1], stride[2]);
    printf("  numel   : %d  (%.2f MB)\n",
           numel(), numel() * (float)sizeof(T) / 1024 / 1024);
    printf("  ptr     : %p  owns=%s\n", data, owns_data ? "yes" : "no (view)");
  }
};
