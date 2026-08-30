#include <benchmark/benchmark.h>
#include <memory>
#include <frontend/circuit_runner.hpp>
#include <frontend/circuit_generators.hpp>

using namespace qlassical::frontend;
using namespace qlassical::middleware;
using namespace qlassical;

static void BM_QFT_HW_Awareness(benchmark::State& state) {
    uint32_t num_qubits = static_cast<uint32_t>(state.range(0));
    bool hw_aware = static_cast<bool>(state.range(1));

    // Setup: generate QFT circuit
    auto circuit = std::make_shared<QuantumCircuit>(make_qft_circuit(num_qubits));

    // Configure transpiler
    TranspilerConfig config;
    config.enable_hw_awareness = hw_aware;
    config.enable_gate_fusion = true; // explicitly enable

    // Benchmark loop
    for (auto _ : state) {
        state.PauseTiming();
        // We re-initialize the runner inside the loop because run() consumes the IRModule
        // To strictly measure the transpilation and execution time:
        auto local_circuit = std::make_shared<QuantumCircuit>(make_qft_circuit(num_qubits));
        CircuitRunner runner(local_circuit, Backend::CPU_OPENMP, config);
        state.ResumeTiming();

        runner.run();
    }
}

// Register benchmark
// Ranges: qubits from 18 to 26, hw_aware (0 or 1)
BENCHMARK(BM_QFT_HW_Awareness)
    ->ArgsProduct({
        benchmark::CreateDenseRange(18, 28, 1),
        {0, 1}
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
