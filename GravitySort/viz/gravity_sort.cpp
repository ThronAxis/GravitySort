// GravitySort — pybind11 C++ module stub
// This wraps the CUDA sort/reduce functions for Python access
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace py = pybind11;

// Forward declarations from CUDA translation units
// (linked via CMake target gravity_sort_py)
// For Kaggle notebook — Python visualization uses pure Python simulation
// This stub provides the module structure; real GPU calls added post-M6

PYBIND11_MODULE(gravity_sort_py, m) {
    m.doc() = "GravitySort Python bindings (pybind11 + CUDA)";

    m.def("get_device_name", []() -> std::string {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        return std::string(prop.name);
    }, "Return the name of the first CUDA device");

    m.def("get_compute_capability", []() -> std::pair<int,int> {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        return {prop.major, prop.minor};
    }, "Return (major, minor) compute capability");

    m.def("get_memory_gb", []() -> float {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        return (float)prop.totalGlobalMem / (1024*1024*1024);
    }, "Return total GPU memory in GB");
}
