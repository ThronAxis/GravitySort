/**
 * GravitySort — Google Benchmark: Sort Variants
 */
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}} while(0)

// External sort functions (link from .cu files)
extern float bitonic_sort_gpu(int*, int);
extern float radix_sort_gpu(uint32_t*, int);

static void BM_StdSort(benchmark::State &state) {
  int N = state.range(0);
  std::vector<int> v(N); std::iota(v.begin(), v.end(), 0);
  std::mt19937 g(42); std::shuffle(v.begin(), v.end(), g);
  for (auto _ : state) {
    auto tmp = v;
    std::sort(tmp.begin(), tmp.end());
    benchmark::DoNotOptimize(tmp);
  }
  state.SetBytesProcessed((int64_t)state.iterations() * N * sizeof(int));
}

static void BM_ThrustSort(benchmark::State &state) {
  int N = state.range(0);
  std::vector<int> h(N); std::iota(h.begin(), h.end(), 0);
  std::mt19937 g(42); std::shuffle(h.begin(), h.end(), g);
  thrust::device_vector<int> d(h);
  cudaDeviceSynchronize();
  for (auto _ : state) {
    thrust::sort(d.begin(), d.end());
    cudaDeviceSynchronize();
  }
  state.SetBytesProcessed((int64_t)state.iterations() * N * sizeof(int));
}

BENCHMARK(BM_StdSort)   ->RangeMultiplier(4)->Range(1<<20, 1<<28)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ThrustSort)->RangeMultiplier(4)->Range(1<<20, 1<<28)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
