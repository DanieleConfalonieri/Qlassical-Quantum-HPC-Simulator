#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <backend/cpu_openmp_engine.hpp>
#include <frontend/frontend.hpp>
#include <cmath>

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
    
    // Assumes GateInstr has a method or flag structure to set arity
    g.flags_arity = arity; 
    
    if (t == GateType::RX || t == GateType::RY || t == GateType::RZ) {
        g.payload.param = param;
    } else if (t == GateType::UNITARY || t == GateType::FUSED_BLOCK) {
        g.payload.matrix_idx = mat_idx;
    }
    
    return g;
}

TEST_CASE("CPUOpenMPEngine single qubit gate kernels", "[backend][kernels]") {
    CPUOpenMPEngine engine(1);
    UnitaryPool pool;
    
    SECTION("Pauli-X Gate") {
        std::vector<GateInstr> prog = { make_gate(GateType::X, 0) };
        engine.execute(prog, pool);
        auto state = engine.state();
        REQUIRE_THAT(std::real(state[0]), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(std::real(state[1]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }
    
    SECTION("Hadamard Gate") {
        std::vector<GateInstr> prog = { make_gate(GateType::H, 0) };
        engine.execute(prog, pool);
        auto state = engine.state();
        double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        REQUIRE_THAT(std::real(state[0]), Catch::Matchers::WithinAbs(inv_sqrt2, 1e-6));
        REQUIRE_THAT(std::real(state[1]), Catch::Matchers::WithinAbs(inv_sqrt2, 1e-6));
    }
    
    SECTION("Pauli-Y Gate") {
        std::vector<GateInstr> prog = { make_gate(GateType::Y, 0) };
        engine.execute(prog, pool);
        auto state = engine.state();
        // Y |0> = i|1>
        REQUIRE_THAT(std::real(state[0]), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(std::imag(state[1]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }
    
    SECTION("Pauli-Z Gate") {
        // Z|0> = |0>, Z|1> = -|1>
        std::vector<GateInstr> prog = { make_gate(GateType::X, 0), make_gate(GateType::Z, 0) };
        engine.execute(prog, pool);
        auto state = engine.state();
        REQUIRE_THAT(std::real(state[1]), Catch::Matchers::WithinAbs(-1.0, 1e-6));
    }
    
    SECTION("S Gate") {
        // S|1> = i|1>
        std::vector<GateInstr> prog = { make_gate(GateType::X, 0), make_gate(GateType::S, 0) };
        engine.execute(prog, pool);
        auto state = engine.state();
        REQUIRE_THAT(std::imag(state[1]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }
}

TEST_CASE("CPUOpenMPEngine parametric single qubit kernels", "[backend][kernels]") {
    CPUOpenMPEngine engine(1);
    UnitaryPool pool;
    
    SECTION("RX Gate") {
        // RX(pi) |0> = -i|1>
        std::vector<GateInstr> prog = { make_gate(GateType::RX, 0, 255, 255, M_PI) };
        engine.execute(prog, pool);
        auto state = engine.state();
        REQUIRE_THAT(std::imag(state[1]), Catch::Matchers::WithinAbs(-1.0, 1e-6));
    }
    
    SECTION("RY Gate") {
        // RY(pi/2) |0> = 1/sqrt(2) (|0> + |1>)
        std::vector<GateInstr> prog = { make_gate(GateType::RY, 0, 255, 255, M_PI / 2.0) };
        engine.execute(prog, pool);
        auto state = engine.state();
        double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        REQUIRE_THAT(std::real(state[0]), Catch::Matchers::WithinAbs(inv_sqrt2, 1e-6));
        REQUIRE_THAT(std::real(state[1]), Catch::Matchers::WithinAbs(inv_sqrt2, 1e-6));
    }
}

TEST_CASE("CPUOpenMPEngine multi qubit kernels (Routing Stress)", "[backend][kernels][routing]") {
    CPUOpenMPEngine engine(3);
    UnitaryPool pool;
    
    SECTION("CX: Control > Target") {
        // CX |100> (q2=1) -> target q0. Result: |101> (idx 5)
        std::vector<GateInstr> prog = { make_gate(GateType::X, 2), make_gate(GateType::CX, 2, 0) };
        engine.execute(prog, pool);
        REQUIRE_THAT(std::real(engine.state()[5]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }
    
    SECTION("CX: Control < Target") {
        // CX |001> (q0=1) -> target q2. Result: |101> (idx 5)
        std::vector<GateInstr> prog = { make_gate(GateType::X, 0), make_gate(GateType::CX, 0, 2) };
        engine.execute(prog, pool);
        REQUIRE_THAT(std::real(engine.state()[5]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }

    SECTION("SWAP: Forward and Backward routing") {
        // |001> -> SWAP(0, 2) -> |100> (idx 4)
        std::vector<GateInstr> prog1 = { make_gate(GateType::X, 0), make_gate(GateType::SWAP, 0, 2) };
        engine.execute(prog1, pool);
        REQUIRE_THAT(std::real(engine.state()[4]), Catch::Matchers::WithinAbs(1.0, 1e-6));
        
        engine.reset();
        
        // |100> -> SWAP(2, 0) -> |001> (idx 1)
        std::vector<GateInstr> prog2 = { make_gate(GateType::X, 2), make_gate(GateType::SWAP, 2, 0) };
        engine.execute(prog2, pool);
        REQUIRE_THAT(std::real(engine.state()[1]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }

    SECTION("CCX: Unordered routing (q1, q0) -> q2") {
        // |011> -> CCX(1,0 -> 2) -> |111> (idx 7)
        std::vector<GateInstr> prog = { 
            make_gate(GateType::X, 0), 
            make_gate(GateType::X, 1), 
            make_gate(GateType::CCX, 1, 0, 2) 
        };
        engine.execute(prog, pool);
        REQUIRE_THAT(std::real(engine.state()[7]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }

    SECTION("CSWAP: Interleaved routing control=1, target=0,2") {
        // CSWAP |011> (q0=1, q1=1, q2=0). ctrl=1, targets=0,2 -> swap q0 and q2 -> |110> (idx 6)
        std::vector<GateInstr> prog = { 
            make_gate(GateType::X, 0), 
            make_gate(GateType::X, 1), 
            make_gate(GateType::CSWAP, 1, 0, 2) 
        };
        engine.execute(prog, pool);
        REQUIRE_THAT(std::real(engine.state()[6]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }
}

TEST_CASE("CPUOpenMPEngine custom unitary kernel (Gather/Scatter Stress)", "[backend][kernels][unitary]") {
    CPUOpenMPEngine engine(3);
    UnitaryPool pool;
    
    SECTION("Apply 2-qubit custom matrix on non-contiguous qubits (q0, q2)") {
        // We define a 4x4 SWAP matrix to stress the Gather/Scatter offsets logic
        std::vector<std::complex<double>> swap_matrix = {
            {1.0, 0}, {0.0, 0}, {0.0, 0}, {0.0, 0}, // |00> -> |00>
            {0.0, 0}, {0.0, 0}, {1.0, 0}, {0.0, 0}, // |01> -> |10>
            {0.0, 0}, {1.0, 0}, {0.0, 0}, {0.0, 0}, // |10> -> |01>
            {0.0, 0}, {0.0, 0}, {0.0, 0}, {1.0, 0}  // |11> -> |11>
        };
        uint32_t idx = pool.store(swap_matrix, 2); 
        
        // Initialize state to |001> (q0=1, q1=0, q2=0) -> idx 1
        // Apply custom SWAP to q0 and q2. 
        // Expected state: |100> (q0=0, q1=0, q2=1) -> idx 4
        std::vector<GateInstr> prog = {
            make_gate(GateType::X, 0),
            make_gate(GateType::UNITARY, 0, 2, 255, 0.0f, idx) 
        };
        
        engine.execute(prog, pool);
        auto state = engine.state();
        
        // Assert old state is cleared and new state is populated
        REQUIRE_THAT(std::real(state[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(std::real(state[4]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }
}

TEST_CASE("CPUOpenMPEngine phase kickback circuit", "[backend][kernels][circuits]") {
    CPUOpenMPEngine engine(2);
    UnitaryPool pool;
    
    SECTION("Phase kickback using CNOT") {
        std::vector<GateInstr> prog = {
            make_gate(GateType::H, 0),       // Q0 in |+>
            make_gate(GateType::X, 1),       // Q1 in |1>
            make_gate(GateType::H, 1),       // Q1 in |->
            make_gate(GateType::CX, 0, 1)    // CNOT control Q0, target Q1
        };
        
        engine.execute(prog, pool);
        auto state = engine.state();
        
        // Expected state: |--> = 0.5 * (|00> - |01> - |10> + |11>)
        // idx 00 -> 0 ->  0.5
        // idx 01 -> 1 -> -0.5 (q0=1, q1=0)
        // idx 10 -> 2 -> -0.5 (q0=0, q1=1)
        // idx 11 -> 3 ->  0.5 (q0=1, q1=1)
        
        REQUIRE_THAT(std::real(state[0]), Catch::Matchers::WithinAbs(0.5, 1e-6));
        REQUIRE_THAT(std::real(state[1]), Catch::Matchers::WithinAbs(-0.5, 1e-6));
        REQUIRE_THAT(std::real(state[2]), Catch::Matchers::WithinAbs(-0.5, 1e-6));
        REQUIRE_THAT(std::real(state[3]), Catch::Matchers::WithinAbs(0.5, 1e-6));
    }
}