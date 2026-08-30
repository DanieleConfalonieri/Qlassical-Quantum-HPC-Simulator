#include <catch2/catch_test_macros.hpp>
#include <frontend/frontend.hpp>
#include <middleware/middleware.hpp>

using namespace qlassical;
using namespace qlassical::middleware;

TEST_CASE("DAG Build and Release (Data Flow)", "[middleware][dag]") {
  SECTION("Empty Module") {
    QuantumCircuit qc(3);
    IRModule module = qc.release();

    DAG dag = DAG::build(std::move(module));
    REQUIRE(dag.nodes.empty());
    REQUIRE(dag.unitary_pool.empty());

    IRModule out_module = std::move(dag).release(module.num_qubits);
    REQUIRE(out_module.empty());
    REQUIRE(out_module.num_qubits == 3);
  }

  SECTION("Standard Module Reconstruction") {
    QuantumCircuit qc(3);
    qc.h(0).cx(0, 1).rx(2, 3.14159).measure({0, 1, 2});
    IRModule module = qc.release();

    std::size_t original_size = module.gate_count();

    DAG dag = DAG::build(std::move(module));
    REQUIRE(dag.nodes.size() == original_size);

    IRModule out_module = std::move(dag).release(module.num_qubits);

    REQUIRE(out_module.gate_count() == original_size);
    REQUIRE(out_module.num_qubits == 3);

    // Check structural equality of instructions
    auto in_prog =
        module.program(); // Note: module was not moved out of in DAG::build,
                          // only its pool was copied/owned
    auto out_prog = out_module.program();

    for (std::size_t i = 0; i < original_size; ++i) {
      REQUIRE(in_prog[i].type == out_prog[i].type);
      REQUIRE(in_prog[i].arity() == out_prog[i].arity());
      for (uint8_t q = 0; q < in_prog[i].arity(); ++q) {
        REQUIRE(in_prog[i].qubits[q] == out_prog[i].qubits[q]);
      }
    }
  }
}

TEST_CASE("Gate Fusion Heuristic Integration",
          "[middleware][fusion_heuristic]") {
  Middleware middleware;

  SECTION("Case A: Should NOT fuse (Only 2 standard gates)") {
    QuantumCircuit qc(2);
    qc.h(0).x(1);
    IRModule module = qc.release();

    std::size_t orig_count = module.gate_count();
    middleware.transpile(module, Backend::CPU_OPENMP);

    REQUIRE(module.gate_count() == orig_count); // Instructions retained

    auto prog = module.program();
    REQUIRE(prog[0].type == GateType::H);
    REQUIRE(prog[1].type == GateType::X);
  }

  SECTION("Case B: Should fuse (3 interconnected gates)") {
    QuantumCircuit qc(2);
    qc.h(0).cx(0, 1).x(1);
    IRModule module = qc.release();

    middleware.transpile(module, Backend::CPU_OPENMP);

    REQUIRE(module.gate_count() == 1); // Fused into one block

    auto prog = module.program();
    REQUIRE(prog[0].type == GateType::FUSED_BLOCK);
    REQUIRE(prog[0].arity() == 2);
  }
}

TEST_CASE("Qubit Windowing: The TLB Miss Hazard",
          "[middleware][qubit_windowing]") {
  QuantumCircuit qc(10);
  qc.x(0).x(1).cx(8, 9);
  IRModule module = qc.release();

  DAG dag = DAG::build(std::move(module));

  // Bypass CPUHwlocTopology and directly instantiate ThresholdCostModel
  // cost_model(2, 1, 0); k_safe = 2, numa_penalty = 1, swap_penalty = 0.
  // With swap_penalty = 0, it aggressively triggers swaps like the greedy model.
  ThresholdCostModel cost_model(2, 1, 0);

  Middleware middleware;
  middleware.pass_qubit_windowing(dag, cost_model, 10);

  IRModule out_module = std::move(dag).release(10);

  // Assertions
  auto prog = out_module.program();

  REQUIRE(prog.size() == 5); // 2 original (X), 2 GLOBAL_SWAP, 1 rewritten CX

  REQUIRE(prog[0].type == GateType::X);
  REQUIRE(prog[0].qubits[0] == 0);

  REQUIRE(prog[1].type == GateType::X);
  REQUIRE(prog[1].qubits[0] == 1);

  // Check injected SWAPs
  REQUIRE(prog[2].type == GateType::GLOBAL_SWAP);
  REQUIRE((prog[2].qubits[0] == 8 || prog[2].qubits[0] == 9));
  REQUIRE((prog[2].qubits[1] == 0 || prog[2].qubits[1] == 1));

  REQUIRE(prog[3].type == GateType::GLOBAL_SWAP);
  REQUIRE((prog[3].qubits[0] == 8 || prog[3].qubits[0] == 9));
  REQUIRE((prog[3].qubits[1] == 0 || prog[3].qubits[1] == 1));
  REQUIRE(prog[2].qubits[0] != prog[3].qubits[0]);
  REQUIRE(prog[2].qubits[1] != prog[3].qubits[1]);

  // Check rewritten CX
  REQUIRE(prog[4].type == GateType::CX);
  // The CX originally targeted 8 and 9, which were mapped to 0 and 1!
  REQUIRE((prog[4].qubits[0] == 0 || prog[4].qubits[0] == 1));
  REQUIRE((prog[4].qubits[1] == 0 || prog[4].qubits[1] == 1));
  REQUIRE(prog[4].qubits[0] != prog[4].qubits[1]);
}
