// -----------------------------
// Qlassical - Frontend.hpp Test
// -----------------------------

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "frontend/frontend.hpp" 

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

using namespace qlassical;

// -------------------------
// GateInstr - Memory layout
// -------------------------

TEST_CASE("GateInstr is exactly 32 bytes and 32-byte aligned", "[GateInstr]") {
    STATIC_REQUIRE(sizeof(GateInstr) == 32);
    STATIC_REQUIRE(alignof(GateInstr) == 32);
}

TEST_CASE("GateInstr field offsets match documented layout", "[GateInstr]") {
    STATIC_REQUIRE(offsetof(GateInstr, type)       == 0);
    STATIC_REQUIRE(offsetof(GateInstr, arity)      == 1);
    STATIC_REQUIRE(offsetof(GateInstr, flags)      == 2);
    STATIC_REQUIRE(offsetof(GateInstr, qubits)     == 4);
    STATIC_REQUIRE(offsetof(GateInstr, params)     == 12);
    STATIC_REQUIRE(offsetof(GateInstr, matrix_idx) == 24);
    STATIC_REQUIRE(offsetof(GateInstr, uid)        == 28);
}

TEST_CASE("GateInstr is trivially copyable and destructible", "[GateInstr]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<GateInstr>);
    STATIC_REQUIRE(std::is_trivially_destructible_v<GateInstr>);
}

TEST_CASE("GateInstr default-initializes to sane values", "[GateInstr]") {
    GateInstr instr{};

    CHECK(instr.arity == 0);
    CHECK(instr.flags == gate_flags::NONE);
    CHECK(instr.qubits[0] == -1);
    CHECK(instr.qubits[1] == -1);
    CHECK(instr.qubits[2] == -1);
    CHECK(instr.qubits[3] == -1);
    CHECK(instr.params[0] == 0.0f);
    CHECK(instr.params[1] == 0.0f);
    CHECK(instr.params[2] == 0.0f);
    CHECK(instr.matrix_idx == NO_MATRIX);
    CHECK(instr.uid == 0);
}

// ----------------------
// SECTION 2: UnitaryPool
// ----------------------

TEST_CASE("UnitaryPool stores and retrieves a 1-qubit matrix", "[pool]") {
    UnitaryPool pool;

    std::vector<std::complex<double>> identity = {
        {1, 0}, {0, 0},
        {0, 0}, {1, 0}
    };

    uint32_t idx = pool.store(identity, 1);

    REQUIRE(idx == 0);
    REQUIRE(pool.num_matrices() == 1);
    REQUIRE(pool.total_elements() == 4);

    auto retrieved = pool.get(idx);
    REQUIRE(retrieved.size() == 4);
    CHECK(retrieved[0] == std::complex<double>(1, 0));
    CHECK(retrieved[1] == std::complex<double>(0, 0));
    CHECK(retrieved[2] == std::complex<double>(0, 0));
    CHECK(retrieved[3] == std::complex<double>(1, 0));
}

TEST_CASE("UnitaryPool stores a 2-qubit matrix correctly", "[pool]") {
    UnitaryPool pool;

    std::vector<std::complex<double>> cnot_matrix(16, {0, 0});
    cnot_matrix[0]  = {1, 0};
    cnot_matrix[5]  = {1, 0};
    cnot_matrix[11] = {1, 0};
    cnot_matrix[14] = {1, 0};

    uint32_t idx = pool.store(cnot_matrix, 2);

    REQUIRE(idx == 0);
    REQUIRE(pool.total_elements() == 16);

    auto retrieved = pool.get(idx);
    REQUIRE(retrieved.size() == 16);
    CHECK(retrieved[0]  == std::complex<double>(1, 0));
    CHECK(retrieved[11] == std::complex<double>(1, 0));
}

TEST_CASE("UnitaryPool assigns sequential indices", "[pool]") {
    UnitaryPool pool;

    std::vector<std::complex<double>> mat1(4, {1, 0});
    std::vector<std::complex<double>> mat2(4, {0, 1});

    uint32_t idx0 = pool.store(mat1, 1);
    uint32_t idx1 = pool.store(mat2, 1);

    CHECK(idx0 == 0);
    CHECK(idx1 == 1);
    CHECK(pool.num_matrices() == 2);
    CHECK(pool.total_elements() == 8);
}

TEST_CASE("UnitaryPool rejects wrong-sized matrix", "[pool]") {
    UnitaryPool pool;
    std::vector<std::complex<double>> bad_matrix(3, {1, 0});
    REQUIRE_THROWS_AS(pool.store(bad_matrix, 1), std::invalid_argument);
}

TEST_CASE("UnitaryPool starts empty", "[pool]") {
    UnitaryPool pool;
    CHECK(pool.empty());
    CHECK(pool.num_matrices() == 0);
    CHECK(pool.total_elements() == 0);
}

// -------------------
// SECTION 3: IRModule
// --------------------

TEST_CASE("IRModule default construction", "[IRModule]") {
    IRModule mod;
    CHECK(mod.num_qubits == 0);
    CHECK(mod.empty());
    CHECK(mod.gate_count() == 0);
    CHECK(mod.unitaries().empty());
}

TEST_CASE("IRModule explicit construction with qubit count and hint", "[IRModule]") {
    IRModule mod(10, 512);
    CHECK(mod.num_qubits == 10);
    CHECK(mod.empty());
    CHECK(mod.gate_count() == 0);
}

TEST_CASE("IRModule is move-only", "[IRModule]") {
    STATIC_REQUIRE(std::is_move_constructible_v<IRModule>);
    STATIC_REQUIRE(std::is_move_assignable_v<IRModule>);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<IRModule>);
    STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<IRModule>);
}

TEST_CASE("IRModule move transfers ownership", "[IRModule]") {
    IRModule a(5, 64);
    a.gate_stream.push_back(GateInstr{});

    REQUIRE(a.gate_count() == 1);

    IRModule b = std::move(a);
    CHECK(b.num_qubits == 5);
    CHECK(b.gate_count() == 1);
}

TEST_CASE("IRModule span accessors return correct views", "[IRModule]") {
    IRModule mod(2);
    GateInstr g{};
    g.type = GateType::H;
    g.arity = 1;
    g.qubits[0] = 0;
    mod.gate_stream.push_back(g);

    const IRModule& cmod = mod;
    auto const_prog = cmod.program();
    REQUIRE(const_prog.size() == 1);
    CHECK(const_prog[0].type == GateType::H);

    auto mut_prog = mod.program();
    REQUIRE(mut_prog.size() == 1);
    mut_prog[0].flags = gate_flags::ADJOINT;
    CHECK(mod.gate_stream[0].flags == gate_flags::ADJOINT);
}

// ----------------------------------------------
// SECTION 4: QuantumCircuit - Single-qubit gates
// ----------------------------------------------

TEST_CASE("QuantumCircuit builds a single H gate", "[builder][1q]") {
    QuantumCircuit qc(1);
    qc.h(0);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::H);
    CHECK(prog[0].arity == 1);
    CHECK(prog[0].qubits[0] == 0);
    CHECK(prog[0].qubits[1] == -1);
}

TEST_CASE("QuantumCircuit builds all single-qubit non-parametric gates", "[builder][1q]") {
    QuantumCircuit qc(1);
    qc.h(0).x(0).y(0).z(0).s(0);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 5);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::H);
    CHECK(prog[1].type == GateType::X);
    CHECK(prog[2].type == GateType::Y);
    CHECK(prog[3].type == GateType::Z);
    CHECK(prog[4].type == GateType::S);

    for (std::size_t i = 0; i < 5; ++i) {
        CAPTURE(i);
        CHECK(prog[i].arity == 1);
        CHECK(prog[i].qubits[0] == 0);
    }
}

// ---------------------------------------------------------
// SECTION 5: QuantumCircuit - Single-qubit parametric gates
// ---------------------------------------------------------

TEST_CASE("QuantumCircuit RX stores angle correctly", "[builder][1q][param]") {
    const double theta = M_PI / 4.0;
    QuantumCircuit qc(1);
    qc.rx(0, theta);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::RX);
    CHECK(prog[0].arity == 1);
    CHECK(prog[0].qubits[0] == 0);
    CHECK_THAT(static_cast<double>(prog[0].params[0]),
               Catch::Matchers::WithinAbs(theta, 1e-6));
    CHECK(prog[0].params[1] == 0.0f);
    CHECK(prog[0].params[2] == 0.0f);
}

TEST_CASE("QuantumCircuit RY and RZ parametric gates", "[builder][1q][param]") {
    QuantumCircuit qc(2);
    qc.ry(0, 1.234).rz(1, -0.567);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 2);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::RY);
    CHECK_THAT(static_cast<double>(prog[0].params[0]),
               Catch::Matchers::WithinAbs(1.234, 1e-6));

    CHECK(prog[1].type == GateType::RZ);
    CHECK(prog[1].qubits[0] == 1);
    CHECK_THAT(static_cast<double>(prog[1].params[0]),
               Catch::Matchers::WithinAbs(-0.567, 1e-6));
}

// -------------------------------------------
// SECTION 6: QuantumCircuit - Two-qubit gates
// --------------------------------------------

TEST_CASE("QuantumCircuit CX (CNOT) gate", "[builder][2q]") {
    QuantumCircuit qc(3);
    qc.cx(0, 2);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::CX);
    CHECK(prog[0].arity == 2);
    CHECK(prog[0].qubits[0] == 0);
    CHECK(prog[0].qubits[1] == 2);
    CHECK(prog[0].qubits[2] == -1);
}

TEST_CASE("QuantumCircuit CNOT alias delegates to CX", "[builder][2q]") {
    QuantumCircuit qc1(2);
    qc1.cx(0, 1);
    auto mod1 = std::move(qc1).release();

    QuantumCircuit qc2(2);
    qc2.cnot(0, 1);
    auto mod2 = std::move(qc2).release();

    REQUIRE(mod1.gate_count() == 1);
    REQUIRE(mod2.gate_count() == 1);

    CHECK(mod1.program()[0].type == mod2.program()[0].type);
}

TEST_CASE("QuantumCircuit SWAP gate", "[builder][2q]") {
    QuantumCircuit qc(4);
    qc.swap(1, 3);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::SWAP);
    CHECK(prog[0].arity == 2);
    CHECK(prog[0].qubits[0] == 1);
    CHECK(prog[0].qubits[1] == 3);
}

// ---------------------------------------------
// SECTION 7: QuantumCircuit - Three-qubit gates
// ---------------------------------------------

TEST_CASE("QuantumCircuit CCX (Toffoli) gate", "[builder][3q]") {
    QuantumCircuit qc(4);
    qc.ccx(0, 1, 2);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::CCX);
    CHECK(prog[0].arity == 3);
    CHECK(prog[0].qubits[0] == 0);
    CHECK(prog[0].qubits[1] == 1);
    CHECK(prog[0].qubits[2] == 2);
    CHECK(prog[0].qubits[3] == -1);
}

TEST_CASE("QuantumCircuit Toffoli alias delegates to CCX", "[builder][3q]") {
    QuantumCircuit qc(3);
    qc.toffoli(0, 1, 2);
    auto mod = std::move(qc).release();
    REQUIRE(mod.gate_count() == 1);
    CHECK(mod.program()[0].type == GateType::CCX);
}

TEST_CASE("QuantumCircuit CSWAP (Fredkin) gate", "[builder][3q]") {
    QuantumCircuit qc(3);
    qc.cswap(0, 1, 2);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::CSWAP);
    CHECK(prog[0].arity == 3);
    CHECK(prog[0].qubits[0] == 0);
    CHECK(prog[0].qubits[1] == 1);
    CHECK(prog[0].qubits[2] == 2);
}

TEST_CASE("QuantumCircuit Fredkin alias delegates to CSWAP", "[builder][3q]") {
    QuantumCircuit qc(3);
    qc.fredkin(0, 1, 2);
    auto mod = std::move(qc).release();
    REQUIRE(mod.gate_count() == 1);
    CHECK(mod.program()[0].type == GateType::CSWAP);
}

// ------------------------------------------
// SECTION 8: QuantumCircuit - Custom unitary
// -------------------------------------------

TEST_CASE("QuantumCircuit applies a custom 1-qubit unitary", "[builder][unitary]") {
    const double s = 1.0 / std::sqrt(2.0);
    std::vector<std::complex<double>> hadamard = {
        {s, 0}, { s, 0},
        {s, 0}, {-s, 0}
    };

    int16_t qubits[] = {0};
    QuantumCircuit qc(2);
    qc.unitary(qubits, hadamard);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::UNITARY);
    CHECK(prog[0].arity == 1);
    CHECK(prog[0].qubits[0] == 0);
    CHECK(prog[0].matrix_idx != NO_MATRIX);

    REQUIRE(mod.unitary_pool.num_matrices() == 1);
    auto stored = mod.unitary_pool.get(prog[0].matrix_idx);
    REQUIRE(stored.size() == 4);
    CHECK_THAT(stored[0].real(), Catch::Matchers::WithinAbs(s, 1e-12));
    CHECK_THAT(stored[3].real(), Catch::Matchers::WithinAbs(-s, 1e-12));
}

TEST_CASE("QuantumCircuit applies a custom 2-qubit unitary", "[builder][unitary]") {
    std::vector<std::complex<double>> identity(16, {0, 0});
    identity[0]  = {1, 0};
    identity[5]  = {1, 0};
    identity[10] = {1, 0};
    identity[15] = {1, 0};

    int16_t qubits[] = {0, 1};
    QuantumCircuit qc(2);
    qc.unitary(qubits, identity);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    CHECK(mod.program()[0].arity == 2);
    CHECK(mod.unitary_pool.total_elements() == 16);
}

TEST_CASE("Custom unitary rejects zero arity", "[builder][unitary][error]") {
    std::vector<std::complex<double>> empty_mat;
    std::vector<int16_t> no_qubits;

    REQUIRE_THROWS_AS(
        QuantumCircuit(2).unitary(no_qubits, empty_mat),
        std::invalid_argument
    );
}

// --------------------------------------------------------
// SECTION 9: QuantumCircuit - Circuit structure directives
// --------------------------------------------------------

TEST_CASE("QuantumCircuit barrier with specific qubits", "[builder][directive]") {
    QuantumCircuit qc(4);
    qc.barrier({0, 1, 2});
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::BARRIER);
    CHECK(prog[0].arity == 3);
    CHECK(prog[0].qubits[0] == 0);
    CHECK(prog[0].qubits[1] == 1);
    CHECK(prog[0].qubits[2] == 2);
}

TEST_CASE("QuantumCircuit barrier with no qubits (global barrier)", "[builder][directive]") {
    QuantumCircuit qc(3);
    qc.barrier();
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    CHECK(mod.program()[0].type == GateType::BARRIER);
    CHECK(mod.program()[0].arity == 0);
}

TEST_CASE("QuantumCircuit barrier clamps to 4 qubits", "[builder][directive]") {
    QuantumCircuit qc(6);
    qc.barrier({0, 1, 2, 3, 4});
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    CHECK(mod.program()[0].arity == 4);
    CHECK(mod.program()[0].qubits[3] == 3);
}

TEST_CASE("QuantumCircuit measure single qubit", "[builder][directive]") {
    QuantumCircuit qc(2);
    qc.measure(0);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 1);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::MEASURE);
    CHECK(prog[0].arity == 1);
    CHECK(prog[0].qubits[0] == 0);
}

TEST_CASE("QuantumCircuit measure multiple qubits", "[builder][directive]") {
    QuantumCircuit qc(3);
    qc.measure({0, 1, 2});
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        CAPTURE(i);
        CHECK(mod.program()[i].type == GateType::MEASURE);
        CHECK(mod.program()[i].qubits[0] == static_cast<int16_t>(i));
    }
}

// -----------------------------------------------
// SECTION 10: QuantumCircuit - composite circuits
// -----------------------------------------------

TEST_CASE("QuantumCircuit builds Bell state circuit", "[builder]") {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).measure({0, 1});
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 4);
    auto prog = mod.program();

    CHECK(prog[0].type == GateType::H);
    CHECK(prog[1].type == GateType::CX);
    CHECK(prog[2].type == GateType::MEASURE);
    CHECK(prog[3].type == GateType::MEASURE);
}

TEST_CASE("QuantumCircuit builds a complex mixed circuit", "[builder]") {
    QuantumCircuit qc(4);
    qc.h(0).h(1).cx(0, 2).rx(1, M_PI / 3.0)
      .barrier({0, 1, 2, 3}).ccx(0, 1, 3).swap(2, 3).measure({0, 1, 2, 3});
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 11);
}

// ----------------------------
// SECTION 11: UID monotonicity
// ----------------------------

TEST_CASE("UIDs are strictly monotonically increasing", "[builder][uid]") {
    QuantumCircuit qc(3);
    qc.h(0).x(1).cx(0, 1).ccx(0, 1, 2).barrier().measure(0);
    auto mod = std::move(qc).release();

    auto prog = mod.program();
    REQUIRE(prog.size() >= 2);

    for (std::size_t i = 1; i < prog.size(); ++i) {
        CAPTURE(i, prog[i - 1].uid, prog[i].uid);
        CHECK(prog[i].uid > prog[i - 1].uid);
    }
}

// -------------------------------------------------------
// SECTION 12: QuantumCircuit - Move semantics & release()
// -------------------------------------------------------

TEST_CASE("QuantumCircuit is move-only", "[builder][move]") {
    STATIC_REQUIRE(std::is_move_constructible_v<QuantumCircuit>);
    STATIC_REQUIRE(std::is_move_assignable_v<QuantumCircuit>);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<QuantumCircuit>);
    STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<QuantumCircuit>);
}

TEST_CASE("QuantumCircuit move construction preserves state", "[builder][move]") {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1);

    QuantumCircuit qc2 = std::move(qc);
    auto mod = std::move(qc2).release();

    CHECK(mod.num_qubits == 3);
    REQUIRE(mod.gate_count() == 2);
    CHECK(mod.program()[0].type == GateType::H);
    CHECK(mod.program()[1].type == GateType::CX);
}

TEST_CASE("QuantumCircuit::release() transfers to IRModule via move", "[builder][move]") {
    QuantumCircuit qc(2);
    qc.h(0).h(1);
    auto mod = std::move(qc).release();

    CHECK(mod.num_qubits == 2);
    REQUIRE(mod.gate_count() == 2);
}

TEST_CASE("QuantumCircuit query methods", "[builder]") {
    QuantumCircuit qc(5);
    CHECK(qc.num_qubits() == 5);
    CHECK(qc.size() == 0);

    qc.h(0).x(1).cx(2, 3);
    CHECK(qc.size() == 3);

    auto prog = qc.program();
    REQUIRE(prog.size() == 3);
    CHECK(prog[0].type == GateType::H);
}

// --------------------------------------------------
// SECTION 13: Qubit validation - boundary conditions
// --------------------------------------------------

TEST_CASE("Out-of-range qubit throws std::out_of_range", "[builder][error]") {
    SECTION("Qubit index too high") {
        REQUIRE_THROWS_AS(QuantumCircuit(2).h(2), std::out_of_range);
    }

    SECTION("Qubit index far too high") {
        REQUIRE_THROWS_AS(QuantumCircuit(3).x(100), std::out_of_range);
    }

    SECTION("Negative qubit index") {
        REQUIRE_THROWS_AS(QuantumCircuit(3).h(-1), std::out_of_range);
    }

    SECTION("Valid qubit at boundary") {
        REQUIRE_NOTHROW(QuantumCircuit(3).h(2));
    }
}

TEST_CASE("Two-qubit gate rejects identical qubits", "[builder][error]") {
    REQUIRE_THROWS_AS(QuantumCircuit(3).cx(1, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(QuantumCircuit(4).swap(2, 2), std::invalid_argument);
}

TEST_CASE("Three-qubit gate rejects non-distinct qubits", "[builder][error]") {
    SECTION("First two equal") {
        REQUIRE_THROWS_AS(QuantumCircuit(4).ccx(0, 0, 1), std::invalid_argument);
    }

    SECTION("First and third equal") {
        REQUIRE_THROWS_AS(QuantumCircuit(4).ccx(1, 0, 1), std::invalid_argument);
    }

    SECTION("Last two equal") {
        REQUIRE_THROWS_AS(QuantumCircuit(4).cswap(0, 2, 2), std::invalid_argument);
    }
}

TEST_CASE("Two-qubit gate with out-of-range qubit throws", "[builder][error]") {
    REQUIRE_THROWS_AS(QuantumCircuit(2).cx(0, 5), std::out_of_range);
    REQUIRE_THROWS_AS(QuantumCircuit(2).cx(5, 0), std::out_of_range);
}

TEST_CASE("Three-qubit gate with out-of-range qubit throws", "[builder][error]") {
    REQUIRE_THROWS_AS(QuantumCircuit(3).ccx(0, 1, 10), std::out_of_range);
}

TEST_CASE("Barrier validates qubit indices", "[builder][error]") {
    REQUIRE_THROWS_AS(QuantumCircuit(2).barrier({0, 5}), std::out_of_range);
}

TEST_CASE("Measure validates qubit index", "[builder][error]") {
    REQUIRE_THROWS_AS(QuantumCircuit(2).measure(3), std::out_of_range);
}

// ----------------------------
// SECTION 14: Helper functions
// ----------------------------

TEST_CASE("gate_arity returns correct arity for standard gates", "[helpers]") {
    CHECK(gate_arity(GateType::H)  == 1);
    CHECK(gate_arity(GateType::X)  == 1);
    CHECK(gate_arity(GateType::RX) == 1);
    CHECK(gate_arity(GateType::RZ) == 1);
    CHECK(gate_arity(GateType::S)  == 1);

    CHECK(gate_arity(GateType::CX)   == 2);
    CHECK(gate_arity(GateType::SWAP) == 2);

    CHECK(gate_arity(GateType::CCX)   == 3);
    CHECK(gate_arity(GateType::CSWAP) == 3);

    CHECK(gate_arity(GateType::UNITARY)     == 0);
    CHECK(gate_arity(GateType::FUSED_BLOCK) == 0);
    CHECK(gate_arity(GateType::BARRIER)     == 0);
    CHECK(gate_arity(GateType::MEASURE)     == 0);
}

TEST_CASE("gate_is_parametric identifies parametric gates", "[helpers]") {
    CHECK(gate_is_parametric(GateType::RX));
    CHECK(gate_is_parametric(GateType::RY));
    CHECK(gate_is_parametric(GateType::RZ));

    CHECK_FALSE(gate_is_parametric(GateType::H));
    CHECK_FALSE(gate_is_parametric(GateType::X));
    CHECK_FALSE(gate_is_parametric(GateType::CX));
    CHECK_FALSE(gate_is_parametric(GateType::CCX));
    CHECK_FALSE(gate_is_parametric(GateType::BARRIER));
}

TEST_CASE("gate_uses_matrix identifies matrix-referencing gates", "[helpers]") {
    CHECK(gate_uses_matrix(GateType::UNITARY));
    CHECK(gate_uses_matrix(GateType::FUSED_BLOCK));

    CHECK_FALSE(gate_uses_matrix(GateType::H));
    CHECK_FALSE(gate_uses_matrix(GateType::CX));
    CHECK_FALSE(gate_uses_matrix(GateType::BARRIER));
    CHECK_FALSE(gate_uses_matrix(GateType::GLOBAL_SWAP));
}

TEST_CASE("gate_is_directive identifies compiler directives", "[helpers]") {
    CHECK(gate_is_directive(GateType::FUSED_BLOCK));
    CHECK(gate_is_directive(GateType::GLOBAL_SWAP));
    CHECK(gate_is_directive(GateType::BARRIER));
    CHECK(gate_is_directive(GateType::MEASURE));

    CHECK_FALSE(gate_is_directive(GateType::H));
    CHECK_FALSE(gate_is_directive(GateType::CX));
    CHECK_FALSE(gate_is_directive(GateType::CCX));
    CHECK_FALSE(gate_is_directive(GateType::UNITARY));
}

// ------------------------------------
// SECTION 15: NO_MATRIX sentinel value
// ------------------------------------

TEST_CASE("NO_MATRIX sentinel is UINT32_MAX", "[ir]") {
    STATIC_REQUIRE(NO_MATRIX == UINT32_MAX);
}

TEST_CASE("Standard gates have matrix_idx == NO_MATRIX", "[ir]") {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).rx(0, 1.0);
    auto mod = std::move(qc).release();

    for (const auto& instr : mod.program()) {
        CHECK(instr.matrix_idx == NO_MATRIX);
    }
}

// ---------------------
// SECTION 16: GateFlags
// ---------------------

TEST_CASE("Gate flags are distinct bit values", "[ir][flags]") {
    CHECK(gate_flags::NONE         == 0x00);
    CHECK(gate_flags::ADJOINT      == 0x01);
    CHECK(gate_flags::FUSED_HEAD   == 0x02);
    CHECK(gate_flags::FUSED_TAIL   == 0x03);
    CHECK(gate_flags::COMPILER_GEN == 0x04);
}

TEST_CASE("Flags default to NONE in builder-produced instructions", "[ir][flags]") {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1);
    auto mod = std::move(qc).release();

    for (const auto& instr : mod.program()) {
        CHECK(instr.flags == gate_flags::NONE);
    }
}

// ----------------------
// SECTION 17: Edge cases
// ----------------------

TEST_CASE("Empty circuit produces empty module", "[builder][edge]") {
    QuantumCircuit qc(5);
    auto mod = std::move(qc).release();

    CHECK(mod.num_qubits == 5);
    CHECK(mod.empty());
    CHECK(mod.gate_count() == 0);
    CHECK(mod.program().empty());
}

TEST_CASE("Single-qubit circuit with qubit 0", "[builder][edge]") {
    QuantumCircuit qc(1);
    qc.h(0).measure(0);
    auto mod = std::move(qc).release();

    REQUIRE(mod.gate_count() == 2);
    CHECK(mod.program()[0].type == GateType::H);
    CHECK(mod.program()[1].type == GateType::MEASURE);
}

TEST_CASE("Large circuit preserves all gates", "[builder][edge]") {
    constexpr int N = 1000;
    QuantumCircuit qc(2, N);

    for (int i = 0; i < N; ++i) {
        qc.h(0);
    }

    auto mod = std::move(qc).release();
    REQUIRE(mod.gate_count() == static_cast<std::size_t>(N));

    for (const auto& instr : mod.program()) {
        CHECK(instr.type == GateType::H);
        CHECK(instr.qubits[0] == 0);
    }
}