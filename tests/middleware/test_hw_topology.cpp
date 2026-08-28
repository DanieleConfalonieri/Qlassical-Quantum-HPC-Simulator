#include <catch2/catch_test_macros.hpp>
#include <middleware/hw_topology.hpp>

using namespace qlassical::middleware;

TEST_CASE("Hardware Topology Detection", "[middleware][hw_topology]") {
    SECTION("CPU Hwloc Initialization and Metrics") {
        // If hwloc is not enabled, this test will fail automatically
        CPUHwlocTopology topology;
        
        std::size_t l1_size = topology.get_l1_cache_size();
        uint32_t numa_nodes = topology.get_numa_nodes_count();
        uint16_t k_safe = topology.get_safe_qubit_limit();

        // Base assumptions on the architecture
        REQUIRE(l1_size >= 16384); 
        REQUIRE(numa_nodes >= 1);  
        
        // Maths: floor(log2(L1 / 16))
        uint16_t expected_k_safe = static_cast<uint16_t>(std::floor(std::log2(l1_size / 16.0)));
        REQUIRE(k_safe == expected_k_safe);
        REQUIRE(k_safe >= 10); // At least 16KB -> k_safe must be >= 10
    }
}