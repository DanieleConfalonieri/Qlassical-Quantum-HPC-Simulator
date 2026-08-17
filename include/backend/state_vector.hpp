#ifndef QLASSICAL_BACKEND_STATE_VECTOR_BASE_HPP
#define QLASSICAL_BACKEND_STATE_VECTOR_BASE_HPP

#include <cstdint>
#include <cstddef>

namespace qlassical::backend {

    class StateVector {
    public:
        virtual ~StateVector() = default;

        // Allocates and initializes the state vector for num_qubits qubits
        virtual void initialize(uint32_t num_qubits) = 0;
        // Allocates and initializes the state vector with the provided amplitudes, checking that is a valid state vector (i.e., the amplitudes are normalized)
        virtual void initialize(std::span<const std::complex<double>> initial_amplitudes) = 0;

        // Query methods
        [[nodiscard]] virtual uint32_t num_qubits() const noexcept = 0;
        [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
        [[nodiscard]] virtual bool empty() const noexcept = 0;
        
    protected:
        StateVector() = default;
        StateVector(const StateVector&) = delete;
        StateVector& operator=(const StateVector&) = delete;
        StateVector(StateVector&&) = default;
        StateVector& operator=(StateVector&&) = default;
    };

} // namespace qlassical::backend

#endif // QLASSICAL_BACKEND_STATE_VECTOR_BASE_HPP
