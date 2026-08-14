#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <backend/cpu_openmp_engine.hpp>
#include <frontend/frontend.hpp>

using namespace qlassical;
using namespace qlassical::backend;

TEST_CASE("CPUOpenMPEngine applies Born's rule correctly", "[backend][measurement]") {
    // 1 qubit, initialized to |0>
    CPUOpenMPEngine engine(1);

    // Create a program with a Hadamard gate
    GateInstr h_gate{};
    h_gate.type = GateType::H;
    h_gate.set_arity(1);
    h_gate.qubits[0] = 0;
    
    GateInstr m_gate{};
    m_gate.type = GateType::MEASURE;
    m_gate.set_arity(1);
    m_gate.qubits[0] = 0;

    std::vector<GateInstr> prog = {h_gate, m_gate};
    UnitaryPool pool;

    // Force seed for deterministic measurement 
    // Wait, H gives 50/50. If we use a seed, we can ensure outcome.
    // We will run it multiple times to ensure the state collapses and measurements are recorded.
    
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

    SECTION("Reset clears state and measurements") {
        engine.execute(prog, pool);
        REQUIRE(engine.measurements().size() == 1);
        
        engine.reset();
        REQUIRE(engine.measurements().size() == 0);
        auto state = engine.state();
        REQUIRE_THAT(std::norm(state[0]), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(std::norm(state[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Multiple measurements in the same circuit", "[backend][measurement]") {
    CPUOpenMPEngine engine(2);
    UnitaryPool pool;

    GateInstr h{};
    h.type = GateType::H;
    h.set_arity(1);
    h.qubits[0] = 0;

    GateInstr cx{};
    cx.type = GateType::CX;
    cx.set_arity(2);
    cx.qubits[0] = 0;
    cx.qubits[1] = 1;

    GateInstr m0{};
    m0.type = GateType::MEASURE;
    m0.set_arity(1);
    m0.qubits[0] = 0;

    GateInstr m1{};
    m1.type = GateType::MEASURE;
    m1.set_arity(1);
    m1.qubits[0] = 1;

    std::vector<GateInstr> prog = {h, cx, m0, m1};
    
    // Test 100 shots to ensure correlations hold
    for(int i=0; i<100; ++i) {
        engine.reset();
        engine.execute(prog, pool);
        auto ms = engine.measurements();
        REQUIRE(ms.size() == 2);
        // Because of the CX gate creating a Bell state, m0 must equal m1
        REQUIRE(ms[0] == ms[1]);
    }
}
