// -------------------------------------------------------------
// Qlassical — Hardware Topology Unit Tests (Catch2)
// -------------------------------------------------------------
//
// Unit tests verifying hwloc topology introspection and NUMA limit calculations.

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <middleware/hw_topology.hpp>

using namespace qlassical::middleware;

TEST_CASE("Hardware Topology Detection", "[middleware][hw_topology]") {
  SECTION("CPU Hwloc Initialization and Metrics") {
    CPUHwlocTopology topology;

    std::size_t numa_memory = topology.get_numa_memory_size_bytes();
    uint32_t numa_nodes = topology.get_numa_nodes_count();
    uint16_t k_safe = topology.get_safe_qubit_limit();

    REQUIRE(numa_memory > 0);
    REQUIRE(numa_nodes >= 1);

    // Verify consistency with the 85% usable memory load factor
    double usable_memory = static_cast<double>(numa_memory) * 0.85;
    uint16_t expected_k_safe =
        static_cast<uint16_t>(std::floor(std::log2(usable_memory / 16.0)));
    REQUIRE(k_safe == expected_k_safe);
  }
}