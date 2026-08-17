#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <backend/cpu_openmp_engine.hpp>
#include <frontend/frontend.hpp>

using namespace qlassical;
using namespace qlassical::backend;

// Helper to easily construct gate instructions
inline GateInstr make_gate(GateType t, uint8_t q0 = 255, uint8_t q1 = 255, uint8_t q2 = 255, float param = 0.0f, uint32_t mat_idx = 0) {
    GateInstr g{};
    g.type = t;
    
    uint8_t arity = 0;
    g.qubits[0] = 255;
    g.qubits[1] = 255;
    g.qubits[2] = 255;
    g.qubits[3] = 255;
    
    if (q0 != 255) { g.qubits[0] = q0; arity = 1; }
    if (q1 != 255) { g.qubits[1] = q1; arity = 2; }
    if (q2 != 255) { g.qubits[2] = q2; arity = 3; }
    
    g.flags_arity = arity; 
    
    if (t == GateType::RX || t == GateType::RY || t == GateType::RZ) {
        g.payload.param = param;
    } else if (t == GateType::UNITARY || t == GateType::FUSED_BLOCK) {
        g.payload.matrix_idx = mat_idx;
    }
    
    return g;
}

TEST_CASE("CPUOpenMPEngine applies Born's rule correctly", "[backend][measurement]") {
    // 1 qubit, initialized to |0>
    CPUOpenMPEngine engine(1);
    UnitaryPool pool;

    std::vector<GateInstr> prog = {
        make_gate(GateType::H, 0),
        make_gate(GateType::MEASURE, 0)
    };
    
    SECTION("Measurement records outcome and collapses state") {
        engine.execute(prog, pool);
        auto ms = engine.measurements();
        REQUIRE(ms.size() == 1);
        
        uint8_t outcome = ms[0];
        auto state = engine.state();
        
        if (outcome == 0) {
            REQUIRE_THAT(std::norm(state[0]), Catch::Matchers::WithinAbs(1.0, 1e-6));
            REQUIRE_THAT(std::norm(state[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));
        } else {
            REQUIRE_THAT(std::norm(state[0]), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(std::norm(state[1]), Catch::Matchers::WithinAbs(1.0, 1e-6));
        }
    }

    SECTION("Deterministic measurement with fixed seed") {
        // Run 1 with seed 42
        engine.set_seed(42);
        engine.execute(prog, pool);
        uint8_t outcome1 = engine.measurements()[0];
        
        // Reset and Run 2 with the same seed
        engine.reset();
        engine.set_seed(42);
        engine.execute(prog, pool);
        uint8_t outcome2 = engine.measurements()[0];
        
        // The PRNG must guarantee the exact same collapse path
        REQUIRE(outcome1 == outcome2);
    }

    SECTION("Reset clears state and measurements") {
        engine.execute(prog, pool);
        REQUIRE(engine.measurements().size() == 1);
        
        engine.reset();
        REQUIRE(engine.measurements().size() == 0);
        
        auto state = engine.state();
        // State should be back to |0>
        REQUIRE_THAT(std::norm(state[0]), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(std::norm(state[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Multiple measurements in the same circuit (Bell State)", "[backend][measurement][entanglement]") {
    CPUOpenMPEngine engine(2);
    UnitaryPool pool;

    std::vector<GateInstr> prog = {
        make_gate(GateType::H, 0),
        make_gate(GateType::CX, 0, 1),
        make_gate(GateType::MEASURE, 0),
        make_gate(GateType::MEASURE, 1)
    };
    
    // Test 100 shots to ensure correlations hold and the PRNG doesn't introduce bias that breaks entanglement logic
    for(int i = 0; i < 100; ++i) {
        engine.reset();
        engine.execute(prog, pool);
        auto ms = engine.measurements();
        
        REQUIRE(ms.size() == 2);
        // Because of the CX gate creating a Bell state, measuring q0 collapses q1 to the exact same logic state.
        REQUIRE(ms[0] == ms[1]);
    }
}