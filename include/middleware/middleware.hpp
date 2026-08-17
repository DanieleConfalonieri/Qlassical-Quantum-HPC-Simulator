#ifndef QLASSICAL_MIDDLEWARE_HPP
#define QLASSICAL_MIDDLEWARE_HPP

// ----------
// Middleware
// ----------

// The Middleware (Middle-End) acts as the bridge between the Frontend (QuantumCircuit IR builder)
// and the Backend (ExecutionEngine). It performs target-aware circuit transpilation, hardware-level
// optimizations, and instruction stream transformations.
//
// Key Responsibilities:
//   - Target Hardware Awareness:
//       Optimizes gate scheduling and data layout based on the target Backend (e.g., OpenMP shared memory,
//       NUMA domain locality, or distributed MPI). For instance, on multi-core NUMA architectures,
//       it inserts GLOBAL_SWAP directives when amplitude strides exceed local NUMA domain limits,
//       minimizing cross-socket interconnect traffic.
//   - Gate Fusion & Optimization:
//       Aggregates consecutive single- and multi-qubit gates into dense FUSED_BLOCK unitaries
//       to maximize cache reuse and compute intensity during backend dispatch.
//   - Gate Canonicalization & Decomposition:
//       Decomposes high-level or unsupported gates into the native gate set supported by the chosen Backend.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <frontend/frontend.hpp>

namespace qlassical::middleware {

    // --------------------------------------------------
    // Backend - Target Execution Backend Identification
    // --------------------------------------------------
    // Specifies the target architecture for which the IRModule
    // will be transpiled and optimized.

    enum class Backend : uint8_t {
        CPU_OPENMP = 0x00,  // Multi-core shared-memory CPU engine (OpenMP, NUMA-aware)
        MPI        = 0x01,  // Distributed-memory multi-node engine (MPI)
        GPU        = 0x02   // Accelerated GPU execution engine (CUDA/HIP)
    };

    // Semantic alias for compatibility with CircuitRunner / user code
    using BackendType = Backend;

    // -----------------------------------------------------------
    // Middleware - Hardware-Aware Circuit Transpiler and Optimizer
    // -----------------------------------------------------------

    class Middleware {
    public:
        // Construction
        constexpr Middleware() noexcept = default;

        // Move and copy semantics
        Middleware(Middleware&&) noexcept = default;
        Middleware& operator=(Middleware&&) noexcept = default;
        Middleware(const Middleware&) = default;
        Middleware& operator=(const Middleware&) = default;

        ~Middleware() = default;

        // -------------------------
        // Transpilation & Optimizer
        // -------------------------

        // Transpiles the given IRModule for the specified target Backend.
        // Performs target-specific gate decomposition, hardware/NUMA-aware SWAP scheduling,
        // and gate fusion optimizations on the underlying intermediate representation.
        //
        // Parameters:
        //   module:  Reference to the IRModule instance to transpile.
        //   backend: Target execution backend (default: CPU_OPENMP).
        void transpile(IRModule& module, Backend backend = Backend::CPU_OPENMP) {
            // Target-aware transpilation passes based on backend
            switch (backend) {
                case Backend::CPU_OPENMP:
                    // Perform NUMA-aware analysis, gate fusion, and hardware scheduling
                    transpile_cpu_openmp(module);
                    break;
                case Backend::MPI:
                    // Perform distributed chunk partitioning and communication scheduling
                    transpile_mpi(module);
                    break;
                case Backend::GPU:
                    // Perform kernel fusion and device memory layout scheduling
                    transpile_gpu(module);
                    break;
                default:
                    transpile_cpu_openmp(module);
                    break;
            }
        }

    private:
        // ---------------------------------
        // Backend-Specific Transpile Passes
        // ---------------------------------

        void transpile_cpu_openmp([[maybe_unused]] IRModule& module) {
            // TODO: Implement NUMA-aware stride optimization and gate fusion passes
        }

        void transpile_mpi([[maybe_unused]] IRModule& module) {
            // TODO: Implement distributed communication minimization and global SWAP passes
        }

        void transpile_gpu([[maybe_unused]] IRModule& module) {
            // TODO: Implement GPU kernel fusion and memory coalescing optimization passes
        }
    };

} // namespace qlassical::middleware

namespace qlassical {
    // Export middleware types into root qlassical namespace for convenience
    using middleware::Backend;
    using middleware::BackendType;
    using middleware::Middleware;
} // namespace qlassical

#endif // QLASSICAL_MIDDLEWARE_HPP
