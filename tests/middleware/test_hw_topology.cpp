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

    // Verifica la coerenza con il fattore di carico dell'85%
    double usable_memory = static_cast<double>(numa_memory) * 0.85;
    uint16_t expected_k_safe =
        static_cast<uint16_t>(std::floor(std::log2(usable_memory / 16.0)));
    REQUIRE(k_safe == expected_k_safe);
  }
}