#ifndef QLASSICAL_BACKEND_STATE_VECTOR_BASE_HPP
#define QLASSICAL_BACKEND_STATE_VECTOR_BASE_HPP

// -------------------------------------------------------------
// StateVector — Abstract Base Class for Quantum State Vectors
// -------------------------------------------------------------
//
// Defines the common interface for state vector representations across backends.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>

namespace qlassical::backend {

    class StateVector {
    public:
        virtual ~StateVector() = default;

        // --------------------------
        // Initialization & Lifecycle
        // --------------------------

        // Allocates and initializes the state vector for num_qubits qubits to |0...0>.
        virtual void initialize(uint32_t num_qubits) = 0;

        // Allocates and initializes the state vector with the provided amplitudes.
        // Validates that the input is a valid normalized state vector (i.e., sum of |c_i|^2 == 1).
        virtual void initialize(std::span<const std::complex<double>> initial_amplitudes) = 0;

        // -------------
        // Query Methods
        // -------------

        [[nodiscard]] virtual uint32_t num_qubits() const noexcept = 0;
        [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
        [[nodiscard]] virtual bool empty() const noexcept = 0;
        
    protected:
        // Move-only semantics
        StateVector() = default;
        StateVector(const StateVector&) = delete;
        StateVector& operator=(const StateVector&) = delete;
        StateVector(StateVector&&) = default;
        StateVector& operator=(StateVector&&) = default;
    };

} // namespace qlassical::backend

#endif // QLASSICAL_BACKEND_STATE_VECTOR_BASE_HPP
