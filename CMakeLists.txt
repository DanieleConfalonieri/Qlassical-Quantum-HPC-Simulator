cmake_minimum_required(VERSION 3.22)

project(Qlassical
    VERSION 0.1.0
    DESCRIPTION "A quantum circuit simulator for HPC environments"
    LANGUAGES CXX
)

# Set C++ configurations
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build type
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# OpenMP support for parallel backends
find_package(OpenMP REQUIRED)

# hwloc support for hardware awareness
find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(HWLOC hwloc)
endif()

# System Dependencies (Replaces FetchContent)
find_package(Catch2 3 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(benchmark REQUIRED)

# ------------------------
# Qlassical - Core Library
# ------------------------
add_library(qlassical INTERFACE)

target_include_directories(qlassical INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(qlassical INTERFACE
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)

if (HWLOC_FOUND)
    target_compile_definitions(qlassical INTERFACE ENABLE_HWLOC)
    target_include_directories(qlassical INTERFACE ${HWLOC_INCLUDE_DIRS})
    target_link_directories(qlassical INTERFACE ${HWLOC_LIBRARY_DIRS})
    target_link_libraries(qlassical INTERFACE ${HWLOC_LIBRARIES})
endif()

# ------------
# Testing
# ------------
enable_testing()

add_executable(test_frontend tests/frontend/test_frontend.cpp)
target_link_libraries(test_frontend PRIVATE Catch2::Catch2WithMain qlassical)

add_executable(test_backend 
    tests/backend/test_aligned_allocator.cpp
    tests/backend/test_state_vector.cpp
    tests/backend/test_engine_measure.cpp
    tests/backend/test_engine_kernels.cpp
)
target_link_libraries(test_backend PRIVATE Catch2::Catch2WithMain qlassical)

add_executable(test_integration
    tests/integration/algos_no_optimization.cpp
)
target_link_libraries(test_integration PRIVATE Catch2::Catch2WithMain qlassical)

add_executable(test_circuit_runner tests/frontend/test_circuit_runner.cpp)
target_link_libraries(test_circuit_runner PRIVATE Catch2::Catch2WithMain qlassical)

add_executable(test_middleware
    tests/middleware/test_middleware.cpp
    tests/middleware/test_fusion_math.cpp
)
target_link_libraries(test_middleware PRIVATE Catch2::Catch2WithMain qlassical)

add_executable(test_hw_topology
    tests/middleware/test_hw_topology.cpp
)
target_link_libraries(test_hw_topology PRIVATE Catch2::Catch2WithMain qlassical)


# Catch2 test discovery
include(Catch)
catch_discover_tests(test_frontend)
catch_discover_tests(test_backend)
catch_discover_tests(test_integration)
catch_discover_tests(test_circuit_runner)
catch_discover_tests(test_middleware)
catch_discover_tests(test_hw_topology)

# ------------
# Benchmarks
# ------------
add_executable(bench_hw_awareness benchmarks/bench_hw_awareness.cpp)
target_link_libraries(bench_hw_awareness PRIVATE benchmark::benchmark qlassical)
add_executable(bench_scaling benchmarks/bench_scaling.cpp)
target_link_libraries(bench_scaling PRIVATE benchmark::benchmark qlassical)
