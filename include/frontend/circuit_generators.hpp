#ifndef QLASSICAL_FRONTEND_CIRCUIT_GENERATORS_HPP
#define QLASSICAL_FRONTEND_CIRCUIT_GENERATORS_HPP

#include <frontend/frontend.hpp>
#include <cstdint>
#include <cmath>

namespace qlassical::frontend {

    // ---------------------------------------------------------
    // Hardware-Efficient Ansatz (HEA) Generator
    // Generates a parameterized circuit with linear entanglement.
    // ---------------------------------------------------------
    inline QuantumCircuit make_hea_circuit(uint32_t num_qubits, uint32_t depth) {
        // Pre-allocate capacity to prevent costly memory reallocations during circuit construction
        std::size_t gate_hint = num_qubits * depth * 3;
        QuantumCircuit qc(num_qubits, gate_hint);

        // Deterministic dummy parameter for benchmarking consistency
        double theta = 0.1234; 

        for (uint32_t d = 0; d < depth; ++d) {
            // Single-qubit rotation layer
            for (uint32_t q = 0; q < num_qubits; ++q) {
                qc.ry(q, theta);
                qc.rz(q, theta * 1.5);
            }
            
            // Entanglement layer
            for (uint32_t q = 0; q < num_qubits - 1; ++q) {
                qc.cx(q, q + 1);
            }
            
            theta += 0.05;
        }

        return qc;
    }

    // ---------------------------------------------------------
    // Quantum Fourier Transform (QFT) Generator
    // Generates a dense all-to-all connectivity pattern.
    // Ideal for stressing NUMA boundaries and SWAP injection.
    // ---------------------------------------------------------
    inline QuantumCircuit make_qft_circuit(uint32_t num_qubits) {
        // Worst-case capacity hint for nested loops
        QuantumCircuit qc(num_qubits, num_qubits * num_qubits);
        
        for (uint32_t i = 0; i < num_qubits; ++i) {
            qc.h(i);
            for (uint32_t j = i + 1; j < num_qubits; ++j) {
                qc.cx(j, i);
            }
        }
        
        // Final reversal layer
        for (uint32_t i = 0; i < num_qubits / 2; ++i) {
            qc.swap(i, num_qubits - 1 - i);
        }
        
        return qc;
    }
}

#endif // QLASSICAL_FRONTEND_CIRCUIT_GENERATORS_HPP