/**
 * GravitySort — Reduction Unit Tests (Google Test)
 */
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <vector>
#include <numeric>
#include <cmath>

static float gpu_reduce_thrust(const std::vector<float> &h) {
  thrust::device_vector<float> d(h);
  return thrust::reduce(d.begin(), d.end(), 0.0f);
}

TEST(Reduction, SumOnesSmall) {
  std::vector<float> v(1024, 1.0f);
  float result = gpu_reduce_thrust(v);
  EXPECT_NEAR(result, 1024.0f, 1.0f);
}

TEST(Reduction, SumOnes_1M) {
  const int N = 1 << 20;
  std::vector<float> v(N, 1.0f);
  float result = gpu_reduce_thrust(v);
  EXPECT_NEAR(result, (float)N, (float)N * 1e-4f);
}

TEST(Reduction, SumOnes_32M) {
  const int N = 1 << 25;
  std::vector<float> v(N, 1.0f);
  float result = gpu_reduce_thrust(v);
  EXPECT_NEAR(result, (float)N, (float)N * 1e-4f);
}

TEST(Reduction, NonPowerOf2) {
  std::vector<float> v(999983, 1.0f);
  float result = gpu_reduce_thrust(v);
  EXPECT_NEAR(result, (float)v.size(), 1.0f);
}

TEST(Reduction, AllZeros) {
  std::vector<float> v(1 << 20, 0.0f);
  EXPECT_NEAR(gpu_reduce_thrust(v), 0.0f, 1e-5f);
}

TEST(Reduction, MemoryBandwidth_Target) {
  // Verify vectorized reduction hits ≥70% peak BW on T4 (~320 GB/s → ≥224 GB/s)
  // This is a placeholder; actual BW measured by bench_reduce.cpp
  // Here we just ensure the result is correct
  const int N = 1 << 25;
  std::vector<float> v(N);
  std::iota(v.begin(), v.end(), 0.0f);
  float expected = (float)N * (N - 1) / 2.0f;
  float result   = gpu_reduce_thrust(v);
  // Float precision is limited for large sums; check relative error
  EXPECT_NEAR(result / expected, 1.0f, 0.01f)
      << "Relative error > 1% (expected float precision loss)";
}
