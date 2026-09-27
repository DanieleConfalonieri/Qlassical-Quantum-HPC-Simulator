#include <benchmark/benchmark.h>
#include <omp.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>

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

// ==========================================
// A. STRONG SCALING
// Dimensione fissa (26 Qubit), Thread variabili.
// Obiettivo: Trovare il punto di saturazione della memoria.
// ==========================================
static void BM_Strong_Scaling(benchmark::State& state) {
    uint32_t num_qubits = 26; // Fissato a ~1 GB di memoria, sufficiente per stressare la RAM
    int num_threads = static_cast<int>(state.range(0));
    
    omp_set_num_threads(num_threads);

    IRModule module = make_clustered_circuit(num_qubits, 6, 15);
    TranspilerConfig config;
    config.enable_gate_fusion = true;
    config.enable_hw_awareness = true; // Usiamo la best configuration trovata
    
    Middleware middleware;
    middleware.transpile(module, Backend::CPU_OPENMP, config);

    CPUOpenMPEngine engine(num_qubits);

    for (auto _ : state) {
        state.PauseTiming();
        engine.reset();
        state.ResumeTiming();
        
        engine.execute(module);
    }
}

// Range: 1, 2, 4, 8, 16, 32, 64, 112 (Max GPP core su MN5)
BENCHMARK(BM_Strong_Scaling)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(112)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);


// ==========================================
// B. WEAK SCALING
// Carico per thread costante. 
// Ogni raddoppio dei thread raddoppia la memoria (aggiungendo 1 qubit).
// Obiettivo: Verificare che il tempo resti costante.
// ==========================================
static void BM_Weak_Scaling(benchmark::State& state) {
    int shift = static_cast<int>(state.range(0));
    
    // Partiamo da 22 Qubit per 1 Thread (circa 64 MB).
    // shift=0 -> 22Q, 1T
    // shift=1 -> 23Q, 2T
    // shift=2 -> 24Q, 4T ... e così via.
    uint32_t num_qubits = 22 + shift;
    int num_threads = 1 << shift; 

    // Cap a 112 thread se sforiamo la potenza del nodo (es. 128)
    if (num_threads > 112) {
        num_threads = 112; 
    }
    
    omp_set_num_threads(num_threads);

    IRModule module = make_clustered_circuit(num_qubits, 6, 15);
    TranspilerConfig config;
    config.enable_gate_fusion = true;
    config.enable_hw_awareness = true;
    
    Middleware middleware;
    middleware.transpile(module, Backend::CPU_OPENMP, config);

    CPUOpenMPEngine engine(num_qubits);

    for (auto _ : state) {
        state.PauseTiming();
        engine.reset();
        state.ResumeTiming();
        
        engine.execute(module);
    }
}

// shift: da 0 (1 thread) a 6 (64 thread)
BENCHMARK(BM_Weak_Scaling)
    ->DenseRange(0, 6, 1)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();