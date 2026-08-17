#ifndef QLASSICAL_BACKEND_EXECUTION_ENGINE_HPP
#define QLASSICAL_BACKEND_EXECUTION_ENGINE_HPP

#include <complex>
#include <cstdint>
#include <span>
#include <vector>
#include <frontend/frontend.hpp>

namespace qlassical::backend {

    class ExecutionEngine {
    public:
        virtual ~ExecutionEngine() = default;

        // Reset the engine state (e.g. state vector to |0...0> and clear measurements)
        virtual void reset() = 0;

        // Reset to an arbitrary state
        virtual void reset(std::span<const std::complex<double>> initial_state) = 0;

        // Set RNG seed for deterministic measurement outcomes
        virtual void set_seed(uint64_t seed) = 0;

        // Preallocate memory for measurements
        virtual void reserve_measurements(std::size_t count) = 0;

        // Retrieve recorded measurements
        [[nodiscard]] virtual const std::vector<uint8_t>& measurements() const = 0;

        // The program is a span of GateInstr (hot instruction stream).
        // The pool provides runtime matrix data for CUSTOM/FUSED_BLOCK gates.
        // After this call returns, the engine's internal StateVector reflects
        // the result of applying every gate in sequence.
        virtual void execute(std::span<const GateInstr> program,
                             const UnitaryPool& pool) = 0;

        // Convenience overload: execute from an IRModule directly
        void execute(const IRModule& module);

        // State inspection (for measurement, debugging, verification)

        [[nodiscard]] virtual std::span<const std::complex<double>>
        state() const = 0;

        [[nodiscard]] virtual uint32_t num_qubits() const = 0;

    protected:
        ExecutionEngine() = default;
        ExecutionEngine(ExecutionEngine&&) = default;
        ExecutionEngine& operator=(ExecutionEngine&&) = default;

        // No copy strategy
        ExecutionEngine(const ExecutionEngine&) = delete;
        ExecutionEngine& operator=(const ExecutionEngine&) = delete;
    };

} // namespace qlassical::backend

#endif // QLASSICAL_BACKEND_EXECUTION_ENGINE_HPP
