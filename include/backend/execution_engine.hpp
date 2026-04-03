#ifndef QLASSICAL_BACKEND_EXECUTION_ENGINE_HPP
#define QLASSICAL_BACKEND_EXECUTION_ENGINE_HPP

#include <complex>
#include <cstdint>
#include <span>
#include <frontend/frontend.hpp>

namespace qlassical::backend {

    class ExecutionEngine {
    public:
        virtual ~ExecutionEngine() = default;

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
