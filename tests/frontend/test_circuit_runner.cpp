// -------------------------------------------------------------
// Qlassical — CircuitRunner Unit Tests (Catch2)
// -------------------------------------------------------------
//
// Comprehensive unit tests for CircuitRunner orchestration,
// multi-block state vector continuity, and error handling.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "frontend/circuit_runner.hpp"
#include "frontend/frontend.hpp"
#include "middleware/middleware.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

using namespace qlassical;

// -------------------------------------------------------------
// SECTION 1: Constructor & Input Validation (Corner Cases)
// -------------------------------------------------------------

TEST_CASE("CircuitRunner single-circuit valid construction", "[CircuitRunner][constructor]") {
    auto qc = std::make_shared<QuantumCircuit>(2);
    CircuitRunner runner(qc);

    CHECK(runner.num_blocks() == 1);
    CHECK(runner.backend() == Backend::CPU_OPENMP);
    CHECK(runner.circuits().size() == 1);
    CHECK(runner.circuits()[0] == qc);
    CHECK(runner.engine() == nullptr);
}

TEST_CASE("CircuitRunner single-circuit explicit backend construction", "[CircuitRunner][constructor]") {
    auto qc = std::make_shared<QuantumCircuit>(3);
    CircuitRunner runner(qc, Backend::CPU_OPENMP);

    CHECK(runner.backend() == Backend::CPU_OPENMP);
}

TEST_CASE("CircuitRunner single-circuit rejects nullptr", "[CircuitRunner][constructor][error]") {
    std::shared_ptr<QuantumCircuit> null_qc = nullptr;
    REQUIRE_THROWS_AS(CircuitRunner(null_qc), std::invalid_argument);
}

TEST_CASE("CircuitRunner multi-circuit valid construction", "[CircuitRunner][constructor]") {
    auto b1 = std::make_shared<QuantumCircuit>(2);
    auto b2 = std::make_shared<QuantumCircuit>(2);
    auto b3 = std::make_shared<QuantumCircuit>(2);

    std::vector<std::shared_ptr<QuantumCircuit>> blocks = {b1, b2, b3};
    CircuitRunner runner(blocks, Backend::CPU_OPENMP);

    CHECK(runner.num_blocks() == 3);
    CHECK(runner.circuits().size() == 3);
    CHECK(runner.circuits()[0] == b1);
    CHECK(runner.circuits()[1] == b2);
    CHECK(runner.circuits()[2] == b3);
}

TEST_CASE("CircuitRunner multi-circuit rejects empty vector", "[CircuitRunner][constructor][error]") {
    std::vector<std::shared_ptr<QuantumCircuit>> empty_blocks;
    REQUIRE_THROWS_AS(CircuitRunner(empty_blocks), std::invalid_argument);
}

TEST_CASE("CircuitRunner multi-circuit rejects nullptr in block list", "[CircuitRunner][constructor][error]") {
    auto valid_qc = std::make_shared<QuantumCircuit>(2);

    SECTION("Nullptr at index 0") {
        std::vector<std::shared_ptr<QuantumCircuit>> blocks = {nullptr, valid_qc};
        REQUIRE_THROWS_AS(CircuitRunner(blocks), std::invalid_argument);
    }

    SECTION("Nullptr in the middle") {
        std::vector<std::shared_ptr<QuantumCircuit>> blocks = {valid_qc, nullptr, valid_qc};
        REQUIRE_THROWS_AS(CircuitRunner(blocks), std::invalid_argument);
    }

    SECTION("Nullptr at the end") {
        std::vector<std::shared_ptr<QuantumCircuit>> blocks = {valid_qc, valid_qc, nullptr};
        REQUIRE_THROWS_AS(CircuitRunner(blocks), std::invalid_argument);
    }
}

// -------------------------------------------------------------
// SECTION 2: Move Semantics & Type Traits
// -------------------------------------------------------------

TEST_CASE("CircuitRunner is move-only", "[CircuitRunner][move]") {
    STATIC_REQUIRE(std::is_move_constructible_v<CircuitRunner>);
    STATIC_REQUIRE(std::is_move_assignable_v<CircuitRunner>);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<CircuitRunner>);
    STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<CircuitRunner>);
}

TEST_CASE("CircuitRunner move construction preserves state", "[CircuitRunner][move]") {
    auto qc = std::make_shared<QuantumCircuit>(2);
    qc->h(0);

    CircuitRunner runner1(qc);
    CircuitRunner runner2 = std::move(runner1);

    CHECK(runner2.num_blocks() == 1);
    runner2.run();

    CHECK(runner2.num_qubits() == 2);
    CHECK(runner2.state().size() == 4);
}

// -------------------------------------------------------------
// SECTION 3: Pre-Execution State Inspection Guards
// -------------------------------------------------------------

TEST_CASE("Accessing state before run() throws std::runtime_error", "[CircuitRunner][guards]") {
    auto qc = std::make_shared<QuantumCircuit>(2);
    CircuitRunner runner(qc);

    CHECK_THROWS_AS(runner.state(), std::runtime_error);
    CHECK_THROWS_AS(runner.num_qubits(), std::runtime_error);
    CHECK(runner.engine() == nullptr);
}

// -------------------------------------------------------------
// SECTION 4: Single-Block Circuit Execution
// -------------------------------------------------------------

TEST_CASE("Single-block empty circuit (identity / |0> state)", "[CircuitRunner][execution]") {
    auto qc = std::make_shared<QuantumCircuit>(1);
    CircuitRunner runner(qc);
    runner.run();

    CHECK(runner.num_qubits() == 1);
    auto state = runner.state();
    REQUIRE(state.size() == 2);

    CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(1.0, 1e-12));
    CHECK_THAT(state[0].imag(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    CHECK_THAT(state[1].imag(), Catch::Matchers::WithinAbs(0.0, 1e-12));
}

TEST_CASE("Single-block Hadamard gate on 1 qubit", "[CircuitRunner][execution]") {
    auto qc = std::make_shared<QuantumCircuit>(1);
    qc->h(0);

    CircuitRunner runner(qc);
    runner.run();

    auto state = runner.state();
    REQUIRE(state.size() == 2);

    const double expected = 1.0 / std::sqrt(2.0);
    CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
    CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
}

TEST_CASE("Single-block Pauli-X gate on 1 qubit", "[CircuitRunner][execution]") {
    auto qc = std::make_shared<QuantumCircuit>(1);
    qc->x(0);

    CircuitRunner runner(qc);
    runner.run();

    auto state = runner.state();
    REQUIRE(state.size() == 2);

    CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("Single-block self-inverse operations", "[CircuitRunner][execution]") {
    SECTION("H followed by H gives identity |0>") {
        auto qc = std::make_shared<QuantumCircuit>(1);
        qc->h(0).h(0);

        CircuitRunner runner(qc);
        runner.run();

        auto state = runner.state();
        CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(1.0, 1e-6));
        CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    }

    SECTION("X followed by X gives identity |0>") {
        auto qc = std::make_shared<QuantumCircuit>(1);
        qc->x(0).x(0);

        CircuitRunner runner(qc);
        runner.run();

        auto state = runner.state();
        CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(1.0, 1e-6));
        CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Single-block 2-qubit product state H(0) and X(1)", "[CircuitRunner][execution]") {
    auto qc = std::make_shared<QuantumCircuit>(2);
    qc->h(0).x(1);

    CircuitRunner runner(qc);
    runner.run();

    auto state = runner.state();
    REQUIRE(state.size() == 4);

    const double expected = 1.0 / std::sqrt(2.0);
    // State is |10> and |11> (indices 2 and 3)
    CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    CHECK_THAT(state[2].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
    CHECK_THAT(state[3].real(), Catch::Matchers::WithinAbs(expected, 1e-6));

    // Verify normalization
    double sum = 0.0;
    for (const auto& a : state) sum += std::norm(a);
    CHECK_THAT(sum, Catch::Matchers::WithinAbs(1.0, 1e-12));
}

// -------------------------------------------------------------
// SECTION 5: Multi-Block Circuit Execution & State Continuity
// -------------------------------------------------------------

TEST_CASE("Multi-block 2-block execution with zero-copy state transfer", "[CircuitRunner][multiblock]") {
    auto b1 = std::make_shared<QuantumCircuit>(2);
    b1->h(0);

    auto b2 = std::make_shared<QuantumCircuit>(2);
    b2->x(1);

    CircuitRunner runner(std::vector<std::shared_ptr<QuantumCircuit>>{b1, b2});
    runner.run();

    auto state = runner.state();
    REQUIRE(state.size() == 4);

    const double expected = 1.0 / std::sqrt(2.0);
    CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    CHECK_THAT(state[2].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
    CHECK_THAT(state[3].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
}

TEST_CASE("Multi-block 3-block state continuity", "[CircuitRunner][multiblock]") {
    auto b1 = std::make_shared<QuantumCircuit>(2);
    b1->h(0);

    auto b2 = std::make_shared<QuantumCircuit>(2);
    b2->x(1);

    auto b3 = std::make_shared<QuantumCircuit>(2);
    b3->h(0); // Reverses H(0), leaving |10> (index 2)

    CircuitRunner runner(std::vector<std::shared_ptr<QuantumCircuit>>{b1, b2, b3});
    runner.run();

    auto state = runner.state();
    REQUIRE(state.size() == 4);

    CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    CHECK_THAT(state[2].real(), Catch::Matchers::WithinAbs(1.0, 1e-6));
    CHECK_THAT(state[3].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
}

TEST_CASE("Multi-block with empty intermediate block preserves state", "[CircuitRunner][multiblock]") {
    auto b1 = std::make_shared<QuantumCircuit>(1);
    b1->x(0);

    auto b2 = std::make_shared<QuantumCircuit>(1); // empty block

    auto b3 = std::make_shared<QuantumCircuit>(1);
    b3->h(0);

    CircuitRunner runner(std::vector<std::shared_ptr<QuantumCircuit>>{b1, b2, b3});
    runner.run();

    auto state = runner.state();
    REQUIRE(state.size() == 2);

    // H |1> = ( |0> - |1> ) / sqrt(2)
    const double expected = 1.0 / std::sqrt(2.0);
    CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
    CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(-expected, 1e-6));
}

TEST_CASE("Multi-block sequential flipping across 6 blocks", "[CircuitRunner][multiblock]") {
    std::vector<std::shared_ptr<QuantumCircuit>> blocks;
    for (int i = 0; i < 6; ++i) {
        auto b = std::make_shared<QuantumCircuit>(1);
        b->x(0);
        blocks.push_back(b);
    }

    CircuitRunner runner(blocks);
    runner.run();

    // 6 flips -> back to |0>
    auto state = runner.state();
    CHECK_THAT(state[0].real(), Catch::Matchers::WithinAbs(1.0, 1e-6));
    CHECK_THAT(state[1].real(), Catch::Matchers::WithinAbs(0.0, 1e-6));
}

// -------------------------------------------------------------
// SECTION 6: Multi-Block Error Handling (Qubit Mismatch)
// -------------------------------------------------------------

TEST_CASE("Multi-block rejects qubit count mismatch", "[CircuitRunner][multiblock][error]") {
    SECTION("Block 0 has 2 qubits, Block 1 has 3 qubits") {
        auto b1 = std::make_shared<QuantumCircuit>(2);
        auto b2 = std::make_shared<QuantumCircuit>(3);
        CircuitRunner runner(std::vector<std::shared_ptr<QuantumCircuit>>{b1, b2});

        REQUIRE_THROWS_AS(runner.run(), std::invalid_argument);
    }

    SECTION("Block 0 has 4 qubits, Block 1 has 2 qubits") {
        auto b1 = std::make_shared<QuantumCircuit>(4);
        auto b2 = std::make_shared<QuantumCircuit>(2);
        CircuitRunner runner(std::vector<std::shared_ptr<QuantumCircuit>>{b1, b2});

        REQUIRE_THROWS_AS(runner.run(), std::invalid_argument);
    }
}

// -------------------------------------------------------------
// SECTION 7: Backend Management & Unsupported Backends
// -------------------------------------------------------------

TEST_CASE("CircuitRunner handles unimplemented backends appropriately", "[CircuitRunner][backends]") {
    auto qc = std::make_shared<QuantumCircuit>(2);

    SECTION("MPI backend throws runtime_error on run()") {
        CircuitRunner runner(qc, Backend::MPI);
        CHECK(runner.backend() == Backend::MPI);
        REQUIRE_THROWS_AS(runner.run(), std::runtime_error);
    }

    SECTION("GPU backend throws runtime_error on run()") {
        CircuitRunner runner(qc, Backend::GPU);
        CHECK(runner.backend() == Backend::GPU);
        REQUIRE_THROWS_AS(runner.run(), std::runtime_error);
    }
}

// -------------------------------------------------------------
// SECTION 8: Multi-Qubit Scalability & Uniform Superposition
// -------------------------------------------------------------

TEST_CASE("Uniform superposition across 4 qubits (16 states)", "[CircuitRunner][scalability]") {
    auto qc = std::make_shared<QuantumCircuit>(4);
    qc->h(0).h(1).h(2).h(3);

    CircuitRunner runner(qc);
    runner.run();

    auto state = runner.state();
    REQUIRE(state.size() == 16);

    const double expected = 1.0 / 4.0; // 1 / sqrt(16) = 0.25
    double total_prob = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        CAPTURE(i);
        CHECK_THAT(state[i].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
        CHECK_THAT(state[i].imag(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        total_prob += std::norm(state[i]);
    }
    CHECK_THAT(total_prob, Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("Uniform superposition across 6 qubits (64 states)", "[CircuitRunner][scalability]") {
    auto qc = std::make_shared<QuantumCircuit>(6);
    for (int16_t q = 0; q < 6; ++q) {
        qc->h(q);
    }

    CircuitRunner runner(qc);
    runner.run();

    auto state = runner.state();
    REQUIRE(state.size() == 64);

    const double expected = 1.0 / 8.0; // 1 / sqrt(64) = 0.125
    double total_prob = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        CHECK_THAT(state[i].real(), Catch::Matchers::WithinAbs(expected, 1e-6));
        total_prob += std::norm(state[i]);
    }
    CHECK_THAT(total_prob, Catch::Matchers::WithinAbs(1.0, 1e-12));
}

// -------------------------------------------------------------
// SECTION 9: Query & Inspection Methods
// -------------------------------------------------------------

TEST_CASE("CircuitRunner query methods return correct metadata", "[CircuitRunner][query]") {
    auto b1 = std::make_shared<QuantumCircuit>(3);
    auto b2 = std::make_shared<QuantumCircuit>(3);

    CircuitRunner runner(std::vector<std::shared_ptr<QuantumCircuit>>{b1, b2}, Backend::CPU_OPENMP);

    CHECK(runner.num_blocks() == 2);
    CHECK(runner.backend() == Backend::CPU_OPENMP);
    CHECK(runner.circuits().size() == 2);
    CHECK(runner.engine() == nullptr);

    runner.run();

    CHECK(runner.num_qubits() == 3);
    CHECK(runner.engine() != nullptr);
    CHECK(runner.state().size() == 8);
}

// -------------------------------------------------------------
// SECTION 10: Middleware Transpilation
// -------------------------------------------------------------

TEST_CASE("Middleware transpile accepts IRModule and BackendType", "[Middleware][transpile]") {
    Middleware mw;
    IRModule mod(2);

    SECTION("Transpile for CPU_OPENMP backend") {
        REQUIRE_NOTHROW(mw.transpile(mod, Backend::CPU_OPENMP));
    }

    SECTION("Transpile for MPI backend") {
        REQUIRE_NOTHROW(mw.transpile(mod, Backend::MPI));
    }

    SECTION("Transpile for GPU backend") {
        REQUIRE_NOTHROW(mw.transpile(mod, Backend::GPU));
    }

    SECTION("Transpile with default backend argument") {
        REQUIRE_NOTHROW(mw.transpile(mod));
    }

    SECTION("Middleware instance is reusable across different backend types") {
        IRModule mod_cpu(2);
        IRModule mod_mpi(3);
        IRModule mod_gpu(4);

        REQUIRE_NOTHROW(mw.transpile(mod_cpu, Backend::CPU_OPENMP));
        REQUIRE_NOTHROW(mw.transpile(mod_mpi, Backend::MPI));
        REQUIRE_NOTHROW(mw.transpile(mod_gpu, Backend::GPU));
    }

    SECTION("Transpile accepts custom TranspilerConfig with cost model parameters") {
        IRModule mod_custom(4);
        middleware::TranspilerConfig config;
        config.numa_penalty = 20;
        config.swap_penalty = 2;
        config.cost_model_type = middleware::CostModelType::THRESHOLD;
        config.k_safe = 2;

        REQUIRE_NOTHROW(mw.transpile(mod_custom, Backend::CPU_OPENMP, config));
    }
}

TEST_CASE("CircuitRunner accepts and applies custom TranspilerConfig", "[CircuitRunner][transpile][config]") {
    auto qc = std::make_shared<QuantumCircuit>(2);
    qc->h(0).cx(0, 1);

    middleware::TranspilerConfig config;
    config.enable_gate_fusion = true;
    config.enable_hw_awareness = true;
    config.numa_penalty = 15;
    config.swap_penalty = 2;
    config.k_safe = 4;

    CircuitRunner runner(qc, Backend::CPU_OPENMP, config);

    REQUIRE(runner.transpiler_config().numa_penalty == 15);
    REQUIRE(runner.transpiler_config().swap_penalty == 2);
    REQUIRE(runner.transpiler_config().k_safe.value() == 4);

    REQUIRE_NOTHROW(runner.run());

    auto state = runner.state();
    REQUIRE(state.size() == 4);

    // Bell state (|00> + |11>) / sqrt(2)
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    REQUIRE(std::abs(state[0].real() - inv_sqrt2) < 1e-6);
    REQUIRE(std::abs(state[3].real() - inv_sqrt2) < 1e-6);
}

