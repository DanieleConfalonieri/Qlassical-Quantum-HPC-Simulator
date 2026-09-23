#ifndef QLASSICAL_FRONTEND_CIRCUIT_GENERATORS_HPP
#define QLASSICAL_FRONTEND_CIRCUIT_GENERATORS_HPP

#include <frontend/frontend.hpp>
#include <cstdint>
#include <cmath>

namespace qlassical::frontend {

    inline QuantumCircuit make_hea_circuit(uint32_t num_qubits, uint32_t depth) {
        // Stimiamo un numero di porte per evitare riallocazioni
        std::size_t gate_hint = num_qubits * depth * 3;
        QuantumCircuit qc(num_qubits, gate_hint);

        double theta = 0.1234; // Parametro fittizio deterministico

        for (uint32_t d = 0; d < depth; ++d) {
            // Layer di rotazioni
            for (uint32_t q = 0; q < num_qubits; ++q) {
                qc.ry(q, theta);
                qc.rz(q, theta * 1.5);
            }
            
            // Layer di entanglement (chain)
            for (uint32_t q = 0; q < num_qubits - 1; ++q) {
                qc.cx(q, q + 1);
            }
            
            theta += 0.05;
        }

        return qc;
    }

    // Includiamo anche la QFT per futuri test NUMA
    inline QuantumCircuit make_qft_circuit(uint32_t num_qubits) {
        QuantumCircuit qc(num_qubits, num_qubits * num_qubits);
        
        for (uint32_t i = 0; i < num_qubits; ++i) {
            qc.h(i);
            // Approssimazione: usiamo CX al posto delle rotazioni di fase controllate 
            // per semplicità, mantenendo la topologia all-to-all
            for (uint32_t j = i + 1; j < num_qubits; ++j) {
                qc.cx(j, i);
            }
        }
        
        for (uint32_t i = 0; i < num_qubits / 2; ++i) {
            qc.swap(i, num_qubits - 1 - i);
        }
        
        return qc;
    }
}

#endif