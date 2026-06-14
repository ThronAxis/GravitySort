/**
 * GravitySort — Google Benchmark: Sort Variants
 * ─────────────────────────────────────────────
 * Benchmarks:
 *   BM_StdSort       — CPU  std::sort<int>   (baseline)
 *   BM_ThrustSortInt — GPU  thrust::sort<int>
 *   BM_ThrustSortU32 — GPU  thrust::sort<uint32_t>  (matches radix workload)
 */
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstdint>

#define CHECK_CUDA(call) do {                                         \
  cudaError_t e = (call);                                             \
  if (e != cudaSuccess) {                                             \
    fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e)); exit(1);    \
  }} while(0)

// ─── CPU baseline ────────────────────────────────────────────────────────────
static void BM_StdSort(benchmark::State &state) {
  const int N = state.range(0);
  std::vector<int> v(N);
  std::mt19937 g(42);
  std::iota(v.begin(), v.end(), 0);
  std::shuffle(v.begin(), v.end(), g);

  for (auto _ : state) {
    auto tmp = v;
    std::sort(tmp.begin(), tmp.end());
    benchmark::DoNotOptimize(tmp);
  }
  state.SetBytesProcessed((int64_t)state.iterations() * N * sizeof(int));
}

// ─── GPU: thrust::sort<int> ──────────────────────────────────────────────────
static void BM_ThrustSortInt(benchmark::State &state) {
  const int N = state.range(0);
  std::vector<int> h(N);
  std::mt19937 g(42);
  std::iota(h.begin(), h.end(), 0);
  std::shuffle(h.begin(), h.end(), g);

  thrust::device_vector<int> d(h);
  cudaDeviceSynchronize();

  for (auto _ : state) {
    // Re-shuffle on device via copying to avoid sorting already-sorted data
    thrust::copy(h.begin(), h.end(), d.begin());
    cudaDeviceSynchronize();
    thrust::sort(d.begin(), d.end());
    cudaDeviceSynchronize();
  }
  state.SetBytesProcessed((int64_t)state.iterations() * N * sizeof(int));
}

// ─── GPU: thrust::sort<uint32_t>  (matches radix sort workload) ──────────────
static void BM_ThrustSortU32(benchmark::State &state) {
  const int N = state.range(0);
  std::vector<uint32_t> h(N);
  std::mt19937 g(42);
  std::iota(h.begin(), h.end(), 0u);
  std::shuffle(h.begin(), h.end(), g);

  thrust::device_vector<uint32_t> d(h);
  cudaDeviceSynchronize();

  for (auto _ : state) {
    thrust::copy(h.begin(), h.end(), d.begin());
    cudaDeviceSynchronize();
    thrust::sort(d.begin(), d.end());
    cudaDeviceSynchronize();
  }
  state.SetBytesProcessed((int64_t)state.iterations() * N * sizeof(uint32_t));
}

// ─── Registration ─────────────────────────────────────────────────────────────
BENCHMARK(BM_StdSort)
    ->RangeMultiplier(4)->Range(1<<20, 1<<26)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ThrustSortInt)
    ->RangeMultiplier(4)->Range(1<<20, 1<<26)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ThrustSortU32)
    ->RangeMultiplier(4)->Range(1<<20, 1<<26)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
