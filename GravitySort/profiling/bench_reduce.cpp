/**
 * GravitySort — Google Benchmark: Reduction Variants vs Thrust
 */
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <vector>
#include <numeric>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}} while(0)

static float *d_bench = nullptr;
static int    d_bench_n = 0;

static void setup_data(int N) {
  if (d_bench_n != N) {
    if (d_bench) cudaFree(d_bench);
    d_bench_n = N;
    CHECK_CUDA(cudaMalloc(&d_bench, N * sizeof(float)));
    std::vector<float> h(N, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_bench, h.data(), N*sizeof(float), cudaMemcpyHostToDevice));
  }
}

static void BM_ThrustReduce(benchmark::State &state) {
  int N = state.range(0);
  setup_data(N);
  thrust::device_ptr<float> dp(d_bench);
  cudaDeviceSynchronize();
  for (auto _ : state) {
    float r = thrust::reduce(dp, dp+N, 0.0f);
    benchmark::DoNotOptimize(r);
    cudaDeviceSynchronize();
  }
  state.SetBytesProcessed((int64_t)state.iterations() * N * sizeof(float));
}

BENCHMARK(BM_ThrustReduce)->RangeMultiplier(4)->Range(1<<20, 1<<28)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
