// -------------------------------------------------------------
// Qlassical — AlignedAllocator Unit Tests (Catch2)
// -------------------------------------------------------------
//
// Unit tests verifying 64-byte memory alignment guarantees and
// interoperability with standard STL containers (std::vector).

#include <backend/aligned_allocator.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <vector>

using namespace qlassical::backend;

TEST_CASE("AlignedAllocator returns correctly aligned pointers",
          "[backend][allocator]") {
  using CustomAllocator = AlignedAllocator<std::complex<double>, 64>;
  CustomAllocator alloc;

  SECTION("Small allocations are 64-byte aligned") {
    auto *ptr = alloc.allocate(1);
    REQUIRE(ptr != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(ptr) % 64 == 0);
    alloc.deallocate(ptr, 1);
  }

  SECTION("Large allocations are 64-byte aligned") {
    auto *ptr = alloc.allocate(1024);
    REQUIRE(ptr != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(ptr) % 64 == 0);
    alloc.deallocate(ptr, 1024);
  }

  SECTION("Zero allocation returns nullptr") {
    auto *ptr = alloc.allocate(0);
    REQUIRE(ptr == nullptr);
    alloc.deallocate(ptr, 0);
  }
}

TEST_CASE("AlignedAllocator integrates with std::vector",
          "[backend][allocator]") {
  using CustomAllocator = AlignedAllocator<std::complex<double>, 64>;
  std::vector<std::complex<double>, CustomAllocator> vec;

  SECTION("Vector resize and push_back maintains alignment") {
    vec.resize(100);
    REQUIRE(reinterpret_cast<std::uintptr_t>(vec.data()) % 64 == 0);

    vec.push_back({1.0, 0.0});
    REQUIRE(reinterpret_cast<std::uintptr_t>(vec.data()) % 64 == 0);
  }
}
