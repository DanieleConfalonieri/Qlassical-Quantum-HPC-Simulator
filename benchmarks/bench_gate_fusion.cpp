#include <benchmark/benchmark.h>
#include <memory>
#include <frontend/circuit_runner.hpp>
#include <frontend/circuit_generators.hpp>
#include <frontend/stats.hpp> // Se hai implementato la telemetria

using namespace qlassical::frontend;
using namespace qlassical::middleware;
using namespace qlassical;

static void BM_Gate_Fusion(benchmark::State& state) {
    uint32_t num_qubits = static_cast<uint32_t>(state.range(0));
    bool enable_fusion = static_cast<bool>(state.range(1));
    uint32_t depth = 4; // Profondità fissa

    TranspilerConfig config;
    config.enable_gate_fusion = enable_fusion;
    config.enable_hw_awareness = false; // Disabilitato per test su singolo nodo
    config.verbose_logging = false;

    for (auto _ : state) {
        state.PauseTiming();
        // Generazione del circuito fuori dal timing per misurare solo JIT + Esecuzione
        auto circuit = std::make_shared<QuantumCircuit>(make_hea_circuit(num_qubits, depth));
        CircuitRunner runner(circuit, Backend::CPU_OPENMP, config);
        state.ResumeTiming();

        runner.run();
    }
}

// Ranges: qubit da 16 a 24 a step di 2, fusion on/off
BENCHMARK(BM_Gate_Fusion)
    ->ArgsProduct({
        benchmark::CreateDenseRange(16, 24, 2),
        {0, 1}
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();