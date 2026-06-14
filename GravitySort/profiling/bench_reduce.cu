/**
 * GravitySort — Google Benchmark: All Reduction Variants
 * ─────────────────────────────────────────────────────────────────────────────
 * Benchmarks all 4 custom reduction kernels + thrust::reduce baseline.
 * Reports bytes/second to compare directly against T4 peak (320 GB/s).
 *
 *  BM_ReduceNaive      — global atomics        (~10% peak)
 *  BM_ReduceShared     — shared-mem tree       (~50% peak)
 *  BM_ReduceWarp       — __shfl_xor_sync       (~67% peak)
 *  BM_ReduceVec4       — float4 loads          (~85% peak)
 *  BM_ThrustReduce     — thrust::reduce        (reference)
 */
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <vector>
#include <numeric>
#include "reduce_kernels.cuh"

#define CHECK_CUDA(call) do {                                          \
  cudaError_t e = (call);                                              \
  if (e != cudaSuccess) {                                              \
    fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e)); exit(1);     \
  }} while(0)

// ─── Shared device buffer (reused across benchmarks) ─────────────────────────
static float *g_d_buf = nullptr;
static int    g_d_n   = 0;

static void ensure_device_buf(int N) {
  if (g_d_n != N) {
    if (g_d_buf) cudaFree(g_d_buf);
    g_d_n = N;
    CHECK_CUDA(cudaMalloc(&g_d_buf, (size_t)N * sizeof(float)));
    std::vector<float> h(N, 1.0f);
    CHECK_CUDA(cudaMemcpy(g_d_buf, h.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
}

// ─── Output accumulator ───────────────────────────────────────────────────────
static float *g_d_out = nullptr;

static void ensure_out() {
  if (!g_d_out) CHECK_CUDA(cudaMalloc(&g_d_out, sizeof(float)));
}

// ─── Macro to define a benchmark for one kernel variant ───────────────────────
#define DEFINE_REDUCE_BENCH(Name, LaunchFn)                                    \
static void BM_##Name(benchmark::State &state) {                               \
  const int N = state.range(0);                                                \
  ensure_device_buf(N);                                                        \
  ensure_out();                                                                 \
  cudaDeviceSynchronize();                                                      \
  for (auto _ : state) {                                                       \
    CHECK_CUDA(cudaMemset(g_d_out, 0, sizeof(float)));                         \
    LaunchFn(g_d_buf, g_d_out, N);                                             \
    cudaDeviceSynchronize();                                                    \
  }                                                                             \
  state.SetBytesProcessed((int64_t)state.iterations() * N * sizeof(float));   \
}

DEFINE_REDUCE_BENCH(ReduceNaive,  launch_reduce_naive)
DEFINE_REDUCE_BENCH(ReduceShared, launch_reduce_shared)
DEFINE_REDUCE_BENCH(ReduceWarp,   launch_reduce_warp)
DEFINE_REDUCE_BENCH(ReduceVec4,   launch_reduce_vec4)

// ─── Thrust reference ─────────────────────────────────────────────────────────
static void BM_ThrustReduce(benchmark::State &state) {
  const int N = state.range(0);
  ensure_device_buf(N);
  thrust::device_ptr<float> dp(g_d_buf);
  cudaDeviceSynchronize();
  for (auto _ : state) {
    float r = thrust::reduce(dp, dp + N, 0.0f);
    benchmark::DoNotOptimize(r);
    cudaDeviceSynchronize();
  }
  state.SetBytesProcessed((int64_t)state.iterations() * N * sizeof(float));
}

// ─── Registration (N = 1M → 256M floats) ─────────────────────────────────────
#define REG(BM) BENCHMARK(BM)->RangeMultiplier(4)->Range(1<<20, 1<<28)->Unit(benchmark::kMicrosecond)

REG(BM_ReduceNaive);
REG(BM_ReduceShared);
REG(BM_ReduceWarp);
REG(BM_ReduceVec4);
REG(BM_ThrustReduce);

BENCHMARK_MAIN();
