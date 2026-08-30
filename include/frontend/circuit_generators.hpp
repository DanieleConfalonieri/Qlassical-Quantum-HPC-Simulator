#ifndef QLASSICAL_FRONTEND_CIRCUIT_GENERATORS_HPP
#define QLASSICAL_FRONTEND_CIRCUIT_GENERATORS_HPP

#include <cstdint>
#include <cmath>
#include <vector>
#include <numbers>
#include <complex>
#include <frontend/frontend.hpp>

namespace qlassical::frontend {

/**
 * @brief Generates a standard Quantum Fourier Transform (QFT) circuit.
 *
 * @param num_qubits The number of qubits for the QFT circuit.
 * @return An unreleased QuantumCircuit containing the QFT operations.
 */
inline QuantumCircuit make_qft_circuit(uint32_t num_qubits) {
    QuantumCircuit qc(num_qubits);

    for (uint32_t i = 0; i < num_qubits; ++i) {
        // Apply Hadamard on qubit i
        qc.h(static_cast<int16_t>(i));

        // Apply controlled-phase rotations for all j > i
        for (uint32_t j = i + 1; j < num_qubits; ++j) {
            double theta = std::numbers::pi / static_cast<double>(1ULL << (j - i));
            
            // Controlled phase shift unitary (4x4 matrix, row-major)
            // Diagonal: [1, 1, 1, e^{i*theta}]
            std::vector<std::complex<double>> cu_matrix(16, 0.0);
            cu_matrix[0]  = 1.0;
            cu_matrix[5]  = 1.0;
            cu_matrix[10] = 1.0;
            cu_matrix[15] = std::polar(1.0, theta);

            std::vector<int16_t> target_qubits = {
                static_cast<int16_t>(j), 
                static_cast<int16_t>(i)
            };
            qc.unitary(target_qubits, cu_matrix);
        }
    }

    // SWAP sequence to reverse the qubit order back to standard representation
    for (uint32_t i = 0; i < num_qubits / 2; ++i) {
        qc.swap(static_cast<int16_t>(i), static_cast<int16_t>(num_qubits - 1 - i));
    }

    return qc;
}

} // namespace qlassical::frontend

#endif // QLASSICAL_FRONTEND_CIRCUIT_GENERATORS_HPP
