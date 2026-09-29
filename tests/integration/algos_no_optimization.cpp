// -------------------------------------------------------------
// Qlassical — Quantum Algorithm Integration Tests (Catch2)
// -------------------------------------------------------------
//
// End-to-end integration tests simulating complete quantum algorithms:
// Deutsch Algorithm and Quantum Teleportation protocol.

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


TEST_CASE("Quantum Teleportation", "[integration][algos]") {
  // Teleport an arbitrary state |phi> from q0 to q2.
  // We use a state with a complex relative phase to prove phase preservation.
  // |phi> = alpha |0> + beta |1>
  std::complex<double> alpha = {1.0 / std::sqrt(3.0), 0.0};
  std::complex<double> beta  = {0.0, std::sqrt(2.0 / 3.0)};
  
  // Initial state: q0=|phi>, q1=|0>, q2=|0>.
  auto initial_state = std::vector<std::complex<double>>(8, {0.0, 0.0});
  initial_state[0] = alpha; // |000> (q0=0)
  initial_state[1] = beta;  // |001> (q0=1)

  CPUOpenMPEngine engine(initial_state, std::optional<uint64_t>(seed), 2);
  
  // 1. Entanglement and Bell measurement on Alice's side
  QuantumCircuit alice_circuit(3);
  alice_circuit.h(1).cx(1, 2);         // Entangled Bell pair between Alice (q1) and Bob (q2)
  alice_circuit.cx(0, 1).h(0);         // Alice entangles source qubit |phi> (q0) with q1
  alice_circuit.measure(0).measure(1); // Alice measures q0 and q1
  
  // 2. Partial measurement and state collapse
  engine.execute(alice_circuit.release());
  
  auto ms = engine.measurements();
  REQUIRE(ms.size() == 2);
  uint8_t m0 = ms[0];
  uint8_t m1 = ms[1];
  
  // 3. Bob's conditional correction operations
  QuantumCircuit bob_circuit(3);
  if (m1 == 1) bob_circuit.x(2);
  if (m0 == 1) bob_circuit.z(2);
  
  // Execute Bob's circuit on the same engine, continuing the simulation
  engine.execute(bob_circuit.release());
  
  // 4. Verification of teleported state on Bob's qubit (q2)
  // q0 and q1 collapsed to m0 and m1. q2 should perfectly contain |phi>.
  // Final expected computational basis states:
  // |0 m1 m0> -> Index: (0 * 4) + (m1 * 2) + m0
  // |1 m1 m0> -> Index: (1 * 4) + (m1 * 2) + m0
  std::size_t idx_0 = (m1 << 1) | m0;
  std::size_t idx_1 = 4 | (m1 << 1) | m0;
  
  auto state = engine.state();
  
  // Verify probabilities
  REQUIRE_THAT(std::norm(state[idx_0]), Catch::Matchers::WithinAbs(std::norm(alpha), 1e-6));
  REQUIRE_THAT(std::norm(state[idx_1]), Catch::Matchers::WithinAbs(std::norm(beta), 1e-6));
  
  // Verify that the relative complex phase was preserved
  auto ratio = state[idx_1] / state[idx_0];
  auto expected_ratio = beta / alpha;
  REQUIRE_THAT(std::real(ratio), Catch::Matchers::WithinAbs(std::real(expected_ratio), 1e-6));
  REQUIRE_THAT(std::imag(ratio), Catch::Matchers::WithinAbs(std::imag(expected_ratio), 1e-6));
  
  // Verify absolute collapse (all other amplitudes must be 0)
  for (std::size_t i = 0; i < 8; ++i) {
      if (i != idx_0 && i != idx_1) {
          REQUIRE_THAT(std::norm(state[i]), Catch::Matchers::WithinAbs(0.0, 1e-6));
      }
  }
}
