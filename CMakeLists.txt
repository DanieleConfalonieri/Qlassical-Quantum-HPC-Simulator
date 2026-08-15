cmake_minimum_required(VERSION 3.24)

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

# ------------------------
# Qlassical - Core Library
# ------------------------
add_library(qlassical INTERFACE)

target_include_directories(qlassical INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ------------
# Dependencies 
# ------------

include(FetchContent) # automatically downloads Catch2 for testing

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2 
)

FetchContent_MakeAvailable(Catch2)

# Catch2 v3 provides its CMake helpers here
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen3)

target_link_libraries(qlassical INTERFACE Eigen3::Eigen)

# Test configuration
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

include(Catch)
catch_discover_tests(test_frontend)
catch_discover_tests(test_backend)