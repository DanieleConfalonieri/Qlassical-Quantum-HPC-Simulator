#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <backend/cpu_state_vector.hpp>

using namespace qlassical::backend;

TEST_CASE("CPUStateVector initialization and reset", "[backend][statevector]") {
    CPUStateVector sv;
    
    SECTION("Initialization configures correct size and dimension") {
        sv.initialize(3); // 3 qubits
        REQUIRE(sv.num_qubits() == 3);
        REQUIRE(sv.dimension() == 8);
        REQUIRE(!sv.empty());
    }

    SECTION("Initialization sets |0...0> correctly") {
        sv.initialize(2); // 2 qubits, dim 4
        auto amps = sv.amplitudes();
        
        REQUIRE_THAT(std::real(amps[0]), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(std::imag(amps[0]), Catch::Matchers::WithinAbs(0.0, 1e-6));
        
        for (std::size_t i = 1; i < sv.dimension(); ++i) {
            REQUIRE_THAT(std::real(amps[i]), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(std::imag(amps[i]), Catch::Matchers::WithinAbs(0.0, 1e-6));
        }
    }

    SECTION("Re-initialization (reset) clears polluted state") {
        sv.initialize(1);
        sv.data()[1] = {1.0, 0.0}; // manually pollute state
        
        // Resetting
        sv.initialize(1);
        auto amps = sv.amplitudes();
        REQUIRE_THAT(std::real(amps[0]), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(std::real(amps[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));
    }

    SECTION("Data pointers are correctly 64-byte aligned") {
        sv.initialize(4);
        REQUIRE(reinterpret_cast<std::uintptr_t>(sv.data()) % 64 == 0);
    }
}
