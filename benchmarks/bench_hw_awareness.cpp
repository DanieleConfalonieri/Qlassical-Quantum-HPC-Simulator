#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <frontend/circuit_runner.hpp>
#include <frontend/circuit_generators.hpp>
#include <middleware/middleware.hpp>
#include <backend/cpu_openmp_engine.hpp>

using namespace qlassical::frontend;
using namespace qlassical::middleware;
using namespace qlassical::backend;
using namespace qlassical;

IRModule make_clustered_circuit(uint32_t num_qubits, uint32_t cluster_size, uint32_t depth_per_cluster) {
    IRModule module(num_qubits);
    for (uint32_t start_q = 0; start_q < num_qubits; start_q += cluster_size) {
        uint32_t end_q = std::min(start_q + cluster_size, num_qubits);
        for (uint32_t d = 0; d < depth_per_cluster; ++d) {
            for (uint32_t q = start_q; q < end_q; ++q) {
                GateInstr h_gate;
                h_gate.type = GateType::H;
                h_gate.set_arity(1);
                h_gate.qubits[0] = static_cast<uint16_t>(q);
                module.gate_stream.push_back(h_gate);
            }
            for (uint32_t q = start_q; q < end_q - 1; ++q) {
                GateInstr cx_gate;
                cx_gate.type = GateType::CX;
                cx_gate.set_arity(2);
                cx_gate.qubits[0] = static_cast<uint16_t>(q);
                cx_gate.qubits[1] = static_cast<uint16_t>(q + 1);
                module.gate_stream.push_back(cx_gate);
            }
        }
    }
    return module;
}

// Define 4 strategies
enum class OptStrategy { 
    BASELINE = 0, 
    SWAP_ONLY = 1, 
    FUSION_ONLY = 2, 
    SWAP_AND_FUSION = 3 
};

static void BM_Thesis_Matrix(benchmark::State& state) {
    uint32_t num_qubits = static_cast<uint32_t>(state.range(0));
    OptStrategy strategy = static_cast<OptStrategy>(state.range(1));

    uint32_t cluster_size = 6; 
    uint32_t depth = 15;

    IRModule module = make_clustered_circuit(num_qubits, cluster_size, depth);
    std::size_t orig_size = module.program().size();

    TranspilerConfig config;
    config.enable_hw_awareness = (strategy == OptStrategy::SWAP_ONLY || strategy == OptStrategy::SWAP_AND_FUSION);
    config.enable_gate_fusion = (strategy == OptStrategy::FUSION_ONLY || strategy == OptStrategy::SWAP_AND_FUSION);
    
    Middleware middleware;
    middleware.transpile(module, Backend::CPU_OPENMP, config);

    // Telemetry for the final output
    if (num_qubits == 28) {
        std::string s_name;
        if (strategy == OptStrategy::BASELINE) s_name = "BASELINE     ";
        else if (strategy == OptStrategy::SWAP_ONLY) s_name = "SWAP_ONLY    ";
        else if (strategy == OptStrategy::FUSION_ONLY) s_name = "FUSION_ONLY  ";
        else s_name = "SWAP+FUSION  ";

        std::cout << "[TELEMETRY - " << s_name << "] Gates: " 
                  << orig_size << " -> " << module.program().size() << "\n";
    }

    CPUOpenMPEngine engine(num_qubits);

    for (auto _ : state) {
        state.PauseTiming();
        engine.reset();
        state.ResumeTiming();
        engine.execute(module);
    }
}

BENCHMARK(BM_Thesis_Matrix)
    ->ArgsProduct({
        benchmark::CreateDenseRange(24, 28, 1),
        {0, 1, 2, 3} 
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

