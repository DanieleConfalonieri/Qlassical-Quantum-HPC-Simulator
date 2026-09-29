#ifndef QLASSICAL_BACKEND_EXECUTION_ENGINE_HPP
#define QLASSICAL_BACKEND_EXECUTION_ENGINE_HPP

// -----------------------------------------------------------------
// ExecutionEngine — Abstract Base Class for Quantum Execution Engine
// -----------------------------------------------------------------
//
// Base interface for all backend simulation engines (CPU OpenMP, MPI, GPU).
// Executes an IRModule instruction stream against a simulated quantum state.

#include <complex>
#include <cstdint>
#include <span>
#include <vector>

#include <frontend/frontend.hpp>

namespace qlassical::backend {

    class ExecutionEngine {
    public:
        virtual ~ExecutionEngine() = default;

        // --------------------------
        // Lifecycle & Reset Methods
        // --------------------------

        // Reset the engine state (e.g. state vector to |0...0> and clear measurements).
        virtual void reset() = 0;

        // Reset to an arbitrary state vector (must be normalized).
        virtual void reset(std::span<const std::complex<double>> initial_state) = 0;

        // ---------------------
        // Measurement Interface
        // ---------------------

        // Set RNG seed for deterministic measurement outcomes.
        virtual void set_seed(uint64_t seed) = 0;

        // Preallocate memory for measurements.
        virtual void reserve_measurements(std::size_t count) = 0;

        // Retrieve recorded measurements.
        [[nodiscard]] virtual const std::vector<uint8_t>& measurements() const = 0;

        // -------------------
        // Execution Interface
        // -------------------

        // Executes a linear stream of GateInstr using the provided UnitaryPool.
        // The program is a span of GateInstr (hot instruction stream).
        // The pool provides runtime matrix data for UNITARY and FUSED_BLOCK gates.
        // After completion, the internal state reflects all sequential gate transformations.
        virtual void execute(std::span<const GateInstr> program,
                             const UnitaryPool& pool) = 0;

        // Convenience overload: execute directly from an IRModule.
        void execute(const IRModule& module);

        // ----------------
        // State Inspection
        // ----------------

        // Read-only span view of the state vector amplitudes.
        [[nodiscard]] virtual std::span<const std::complex<double>>
        state() const = 0;

        // Total number of qubits in the simulated system.
        [[nodiscard]] virtual uint32_t num_qubits() const = 0;

    protected:
        // Move-only semantics (no copy strategy)
        ExecutionEngine() = default;
        ExecutionEngine(ExecutionEngine&&) = default;
        ExecutionEngine& operator=(ExecutionEngine&&) = default;
        ExecutionEngine(const ExecutionEngine&) = delete;
        ExecutionEngine& operator=(const ExecutionEngine&) = delete;
    };

} // namespace qlassical::backend

#endif // QLASSICAL_BACKEND_EXECUTION_ENGINE_HPP
