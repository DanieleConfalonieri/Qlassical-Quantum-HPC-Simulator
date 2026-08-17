#include <backend/cpu_openmp_engine.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <frontend/frontend.hpp>

using namespace qlassical;
using namespace qlassical::backend;

uint64_t seed = 37; // Fixed seed for reproducibility

TEST_CASE("Deutsch Algorithm", "[integration][algos]") {
  // Initial state |10> (q0=x=0, q1=y=1) -> index 2 is 1.0
  auto initial_state = std::vector<std::complex<double>>{
      {0.0, 0.0}, // |00>
      {0.0, 0.0}, // |01>
      {1.0, 0.0}, // |10>
      {0.0, 0.0}  // |11>
  };

  // Helper lambda for the four oracles
  auto run_deutsch_and_verify = [&](auto apply_oracle, bool is_constant) {
    // A fresh engine is created for each run with the |10> initial state
    CPUOpenMPEngine engine(initial_state, std::optional<uint64_t>(seed), 1);
    QuantumCircuit circuit(2);

    // 1. Prepare superposition for both x (q0) and y (q1)
    circuit.h(0).h(1);

    // 2. Apply the specific Oracle U_f
    apply_oracle(circuit);

    // 3. Interference on x (q0)
    circuit.h(0);

    // Execute
    IRModule prog = circuit.release();
    engine.execute(prog);

    auto state = engine.state();
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    if (is_constant) {
      // For constant functions, x (q0) must be 0.
      // Final state: +/- |0>|-> = +/- 1/sqrt(2) (|00> - |10>)
      // Index 0 = |00>, Index 2 = |10>
      REQUIRE_THAT(std::norm(state[0]), Catch::Matchers::WithinAbs(0.5, 1e-6));
      REQUIRE_THAT(std::norm(state[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));
      REQUIRE_THAT(std::norm(state[2]), Catch::Matchers::WithinAbs(0.5, 1e-6));
      REQUIRE_THAT(std::norm(state[3]), Catch::Matchers::WithinAbs(0.0, 1e-6));
    } else {
      // For balanced functions, x (q0) must be 1.
      // Final state: +/- |1>|-> = +/- 1/sqrt(2) (|01> - |11>)
      // Index 1 = |01>, Index 3 = |11>
      REQUIRE_THAT(std::norm(state[0]), Catch::Matchers::WithinAbs(0.0, 1e-6));
      REQUIRE_THAT(std::norm(state[1]), Catch::Matchers::WithinAbs(0.5, 1e-6));
      REQUIRE_THAT(std::norm(state[2]), Catch::Matchers::WithinAbs(0.0, 1e-6));
      REQUIRE_THAT(std::norm(state[3]), Catch::Matchers::WithinAbs(0.5, 1e-6));
    }
  };

  SECTION("f(x) = 0 (Constant)") {
    run_deutsch_and_verify([](QuantumCircuit& qc) {
      // Identity oracle does nothing
    }, true);
  }

  SECTION("f(x) = 1 (Constant)") {
    run_deutsch_and_verify([](QuantumCircuit& qc) {
      // Oracle flips y (q1)
      qc.x(1);
    }, true);
  }

  SECTION("f(x) = x (Balanced)") {
    run_deutsch_and_verify([](QuantumCircuit& qc) {
      // Oracle flips y (q1) if x (q0) is 1 -> CNOT(q0, q1)
      qc.cx(0, 1);
    }, false);
  }

  SECTION("f(x) = NOT x (Balanced)") {
    run_deutsch_and_verify([](QuantumCircuit& qc) {
      // Oracle flips y (q1) if x (q0) is 0
      // Apply X on q0, then CNOT(q0, q1), then X on q0 to revert
      qc.cx(0, 1).x(0);
    }, false);
  }
}
