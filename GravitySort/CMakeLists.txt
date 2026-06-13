cmake_minimum_required(VERSION 3.24)
project(GravitySort LANGUAGES CXX CUDA)

# ── CUDA Toolkit ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Standards ─────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
# Build only for the detected GPU arch (passed via -DCMAKE_CUDA_ARCHITECTURES=75)
# Default: T4=75. Override with -DCMAKE_CUDA_ARCHITECTURES=75 at configure time.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# ── NVCC flags ────────────────────────────────────────────────────────────────
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# ── Include dirs (global — CUDA headers visible to all targets) ───────────────
include_directories(
  ${PROJECT_SOURCE_DIR}/tensor
  ${PROJECT_SOURCE_DIR}/kernels
  ${PROJECT_SOURCE_DIR}/memory
  ${PROJECT_SOURCE_DIR}/viz
  ${CUDAToolkit_INCLUDE_DIRS}
)

# ── Dependencies ──────────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1)
FetchContent_MakeAvailable(pybind11)

# ── OpenGL (optional — headless Kaggle will skip) ─────────────────────────────
find_package(OpenGL QUIET)
find_package(glfw3  QUIET)
find_package(GLEW   QUIET)
if(OpenGL_FOUND AND glfw3_FOUND AND GLEW_FOUND)
  set(HAS_OPENGL TRUE)
  message(STATUS "OpenGL found — building gravity_viz")
else()
  set(HAS_OPENGL FALSE)
  message(STATUS "OpenGL/GLFW/GLEW not found — gravity_viz skipped (use Python frontend)")
endif()

# ── Helper macro: CUDA executable (nvcc compiles ALL source files) ─────────────
macro(add_cuda_exe target)
  add_executable(${target} ${ARGN})
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cublas)
endmacro()

# ── Sorting kernels ───────────────────────────────────────────────────────────
add_cuda_exe(bitonic_sort  kernels/bitonic.cu)
add_cuda_exe(radix_sort    kernels/radix.cu)
add_cuda_exe(odd_even_sort kernels/odd_even.cu)

# ── Reduction ─────────────────────────────────────────────────────────────────
add_cuda_exe(reduction_demo kernels/reduction.cu)

# ── Memory demos ──────────────────────────────────────────────────────────────
add_cuda_exe(shared_mem_demo memory/shared_mem.cu)
add_cuda_exe(streams_demo    memory/streams.cu)

# ── Tensor ops ────────────────────────────────────────────────────────────────
add_cuda_exe(tensor_demo tensor/tensor_ops.cu)

# ── Benchmarks (.cu = nvcc, Thrust GPU code requires it) ──────────────────────
add_cuda_exe(bench_sort   profiling/bench_sort.cu)
target_link_libraries(bench_sort   PRIVATE benchmark::benchmark)

add_cuda_exe(bench_reduce profiling/bench_reduce.cu)
target_link_libraries(bench_reduce PRIVATE benchmark::benchmark)

# ── Unit Tests (.cu = nvcc, Thrust GPU code requires it) ──────────────────────
enable_testing()

add_cuda_exe(test_sort   tests/test_sort.cu)
target_link_libraries(test_sort   PRIVATE GTest::gtest_main)

add_cuda_exe(test_reduce tests/test_reduce.cu)
target_link_libraries(test_reduce PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_sort)
gtest_discover_tests(test_reduce)

# ── pybind11 Python module ────────────────────────────────────────────────────
# gravity_sort.cpp uses only host CUDA APIs (cudaGetDeviceProperties)
# Compiled by g++ with CUDA headers visible via include_directories above
pybind11_add_module(gravity_sort_py viz/gravity_sort.cpp)
target_include_directories(gravity_sort_py PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gravity_sort_py PRIVATE CUDA::cudart)

# ── OpenGL + CUDA interop visualizer (optional) ───────────────────────────────
if(HAS_OPENGL)
  add_cuda_exe(gravity_viz viz/gl_interop.cu)
  target_link_libraries(gravity_viz PRIVATE
    OpenGL::GL glfw GLEW::GLEW)
  target_compile_definitions(gravity_viz PRIVATE HAS_OPENGL=1)
  message(STATUS "gravity_viz target added")
endif()
