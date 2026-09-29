// -------------------------------------------------------------
// Qlassical — Middleware Optimization Passes Unit Tests (Catch2)
// -------------------------------------------------------------
//
// Tests for DAG dependency construction, gate fusion heuristics,
// and NUMA-aware qubit windowing SWAP injection.

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

TEST_CASE("TranspilerConfig Defaults and Getters", "[middleware][config]") {
  TranspilerConfig config;

  REQUIRE(config.enable_gate_fusion == true);
  REQUIRE(config.enable_hw_awareness == true);
  REQUIRE(config.cost_model_type == CostModelType::THRESHOLD);
  REQUIRE(config.numa_penalty == 10);
  REQUIRE(config.swap_penalty == 1);
  REQUIRE_FALSE(config.k_safe.has_value());
  REQUIRE(config.custom_cost_model == nullptr);

  ThresholdCostModel model(16, 8, 4);
  REQUIRE(model.get_k_safe() == 16);
  REQUIRE(model.get_numa_penalty() == 8);
  REQUIRE(model.get_swap_penalty() == 4);

  GreedyCostModel greedy(12);
  REQUIRE(greedy.get_k_safe() == 12);
  REQUIRE(greedy.get_numa_penalty() == 1);
  REQUIRE(greedy.get_swap_penalty() == 0);
}

TEST_CASE("TranspilerConfig Penalty Tuning via transpile()",
          "[middleware][config][penalties]") {
  Middleware middleware;

  SECTION("High swap penalty suppresses SWAP injection") {
    QuantumCircuit qc(10);
    qc.x(0).x(1).cx(8, 9);
    IRModule module = qc.release();

    TranspilerConfig config;
    config.enable_gate_fusion = false;
    config.enable_hw_awareness = true;
    config.k_safe = 2;
    config.numa_penalty = 1;
    config.swap_penalty = 100; // retention = 1 * 1 < 100 -> No swaps

    middleware.transpile(module, Backend::CPU_OPENMP, config);

    REQUIRE(module.gate_count() == 3);
    auto prog = module.program();
    REQUIRE(prog[0].type == GateType::X);
    REQUIRE(prog[1].type == GateType::X);
    REQUIRE(prog[2].type == GateType::CX);
    REQUIRE(prog[2].qubits[0] == 8);
    REQUIRE(prog[2].qubits[1] == 9);
  }

  SECTION("Low swap penalty triggers SWAP injection") {
    QuantumCircuit qc(10);
    qc.x(0).x(1).cx(8, 9);
    IRModule module = qc.release();

    TranspilerConfig config;
    config.enable_gate_fusion = false;
    config.enable_hw_awareness = true;
    config.k_safe = 2;
    config.numa_penalty = 10;
    config.swap_penalty = 1; // retention = 1 * 10 >= 1 -> Swaps injected

    middleware.transpile(module, Backend::CPU_OPENMP, config);

    REQUIRE(module.gate_count() == 5); // 2 X, 2 GLOBAL_SWAP, 1 CX
    auto prog = module.program();
    REQUIRE(prog[2].type == GateType::GLOBAL_SWAP);
    REQUIRE(prog[3].type == GateType::GLOBAL_SWAP);
    REQUIRE(prog[4].type == GateType::CX);
    REQUIRE((prog[4].qubits[0] == 0 || prog[4].qubits[0] == 1));
    REQUIRE((prog[4].qubits[1] == 0 || prog[4].qubits[1] == 1));
  }
}

TEST_CASE("TranspilerConfig Cost Model Selection (Greedy)",
          "[middleware][config][greedy]") {
  QuantumCircuit qc(10);
  qc.x(0).x(1).cx(8, 9);
  IRModule module = qc.release();

  TranspilerConfig config;
  config.enable_gate_fusion = false;
  config.enable_hw_awareness = true;
  config.cost_model_type = CostModelType::GREEDY;
  config.k_safe = 2;

  Middleware middleware;
  middleware.transpile(module, Backend::CPU_OPENMP, config);

  REQUIRE(module.gate_count() == 5);
  auto prog = module.program();
  REQUIRE(prog[2].type == GateType::GLOBAL_SWAP);
  REQUIRE(prog[3].type == GateType::GLOBAL_SWAP);
  REQUIRE(prog[4].type == GateType::CX);
}

namespace {
class CustomSpyCostModel : public WindowingCostModel {
public:
  mutable bool evaluated = false;
  uint16_t safe_limit;

  explicit CustomSpyCostModel(uint16_t limit) : safe_limit(limit) {}

  uint16_t get_k_safe() const override { return safe_limit; }

  std::optional<std::pair<uint16_t, uint16_t>>
  evaluate_swap([[maybe_unused]] std::span<const uint32_t> window_frequencies,
                [[maybe_unused]] std::span<const uint16_t> log_to_phys,
                [[maybe_unused]] std::span<const uint16_t> phys_to_log) const override {
    evaluated = true;
    return std::nullopt; // Do not inject any swaps
  }
};
} // namespace

TEST_CASE("TranspilerConfig Custom Cost Model Integration",
          "[middleware][config][custom]") {
  Middleware middleware;

  SECTION("Custom cost model is invoked successfully") {
    QuantumCircuit qc(6);
    qc.h(0).cx(4, 5);
    IRModule module = qc.release();

    auto spy = std::make_shared<CustomSpyCostModel>(3);

    TranspilerConfig config;
    config.enable_gate_fusion = false;
    config.enable_hw_awareness = true;
    config.set_cost_model(spy);

    REQUIRE(config.cost_model_type == CostModelType::CUSTOM);
    REQUIRE(config.custom_cost_model == spy);

    middleware.transpile(module, Backend::CPU_OPENMP, config);

    REQUIRE(spy->evaluated == true);
    REQUIRE(module.gate_count() == 2);
  }

  SECTION("Custom cost model type without instance throws exception") {
    QuantumCircuit qc(4);
    qc.h(0);
    IRModule module = qc.release();

    TranspilerConfig config;
    config.cost_model_type = CostModelType::CUSTOM;
    config.custom_cost_model = nullptr;

    REQUIRE_THROWS_AS(middleware.transpile(module, Backend::CPU_OPENMP, config),
                      std::invalid_argument);
  }
}

TEST_CASE("TranspilerConfig k_safe Manual Override",
          "[middleware][config][k_safe]") {
  Middleware middleware;

  QuantumCircuit qc(10);
  qc.x(0).x(1).cx(8, 9);
  IRModule module = qc.release();

  SECTION("k_safe covering all qubits avoids SWAP injection") {
    TranspilerConfig config;
    config.enable_gate_fusion = false;
    config.enable_hw_awareness = true;
    config.k_safe = 16; // All 10 qubits fit in safe zone

    middleware.transpile(module, Backend::CPU_OPENMP, config);
    REQUIRE(module.gate_count() == 3);
  }

  SECTION("k_safe restricting qubits triggers SWAP injection") {
    TranspilerConfig config;
    config.enable_gate_fusion = false;
    config.enable_hw_awareness = true;
    config.k_safe = 2; // Qubits 8 and 9 outside safe zone

    middleware.transpile(module, Backend::CPU_OPENMP, config);
    REQUIRE(module.gate_count() == 5);
  }
}
