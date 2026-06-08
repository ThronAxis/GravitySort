cmake_minimum_required(VERSION 3.24)
project(GravitySort LANGUAGES CXX CUDA)

# ── CUDA / C++ standards ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)   # T4=75, A100=80, RTX30xx=86, RTX40xx=89

# ── Dependencies (header-only or fetched) ───────────────────────────────────
include(FetchContent)

# Google Benchmark
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# pybind11 (optional – for Python frontend)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# ── Global NVCC flags ────────────────────────────────────────────────────────
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# ── Include dirs ─────────────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/tensor)
include_directories(${PROJECT_SOURCE_DIR}/kernels)
include_directories(${PROJECT_SOURCE_DIR}/memory)
include_directories(${PROJECT_SOURCE_DIR}/viz)

# ── OpenGL + CUDA interop (optional) ────────────────────────────────────────
# Requires: libglfw3-dev libglew-dev  (sudo apt install libglfw3-dev libglew-dev)
find_package(OpenGL QUIET)
find_package(glfw3  QUIET)
find_package(GLEW   QUIET)
if(OpenGL_FOUND AND glfw3_FOUND AND GLEW_FOUND)
  set(HAS_OPENGL TRUE)
  message(STATUS "OpenGL found — building gravity_viz (SM heatmap)")
else()
  set(HAS_OPENGL FALSE)
  message(STATUS "OpenGL/GLFW/GLEW not found — gravity_viz skipped (use Python frontend)")
endif()

# ── Helper macro: add a CUDA executable ──────────────────────────────────────
macro(add_cuda_exe target)
  add_executable(${target} ${ARGN})
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_link_libraries(${target} PRIVATE cublas)
endmacro()

# ── Sorting kernels ──────────────────────────────────────────────────────────
add_cuda_exe(bitonic_sort  kernels/bitonic.cu)
add_cuda_exe(radix_sort    kernels/radix.cu)
add_cuda_exe(odd_even_sort kernels/odd_even.cu)

# ── Reduction kernels ────────────────────────────────────────────────────────
add_cuda_exe(reduction_demo kernels/reduction.cu)

# ── Memory demos ─────────────────────────────────────────────────────────────
add_cuda_exe(shared_mem_demo memory/shared_mem.cu)
add_cuda_exe(streams_demo    memory/streams.cu)

# ── Tensor ops ───────────────────────────────────────────────────────────────
add_cuda_exe(tensor_demo tensor/tensor_ops.cu)

# ── Benchmarks ───────────────────────────────────────────────────────────────
add_executable(bench_sort    profiling/bench_sort.cpp)
target_link_libraries(bench_sort    PRIVATE benchmark::benchmark cublas)
set_property(TARGET bench_sort    PROPERTY CXX_STANDARD 17)

add_executable(bench_reduce  profiling/bench_reduce.cpp)
target_link_libraries(bench_reduce  PRIVATE benchmark::benchmark cublas)
set_property(TARGET bench_reduce  PROPERTY CXX_STANDARD 17)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_sort   tests/test_sort.cpp)
target_link_libraries(test_sort   PRIVATE GTest::gtest_main cublas)
add_executable(test_reduce tests/test_reduce.cpp)
target_link_libraries(test_reduce PRIVATE GTest::gtest_main cublas)

include(GoogleTest)
gtest_discover_tests(test_sort)
gtest_discover_tests(test_reduce)

# ── pybind11 Python module ───────────────────────────────────────────────────
pybind11_add_module(gravity_sort_py viz/gravity_sort.cpp)
target_link_libraries(gravity_sort_py PRIVATE cublas)

# ── OpenGL + CUDA interop visualizer ─────────────────────────────────────────
if(HAS_OPENGL)
  add_executable(gravity_viz viz/gl_interop.cu)
  set_target_properties(gravity_viz PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_link_libraries(gravity_viz PRIVATE
    OpenGL::GL
    glfw
    GLEW::GLEW
    cublas
  )
  target_compile_definitions(gravity_viz PRIVATE HAS_OPENGL=1)
  message(STATUS "gravity_viz target added")
endif()
