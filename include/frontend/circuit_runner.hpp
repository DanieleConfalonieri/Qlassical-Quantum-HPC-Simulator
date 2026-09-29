#ifndef QLASSICAL_FRONTEND_CIRCUIT_RUNNER_HPP
#define QLASSICAL_FRONTEND_CIRCUIT_RUNNER_HPP

// -----------------------------------------------------------------
// CircuitRunner — High-Level Quantum Circuit Execution Orchestrator
// -----------------------------------------------------------------
//
// High-level execution orchestrator for quantum circuits.
// Encapsulates the complete execution pipeline across the three architectural layers:
//   1. FrontEnd: Circuit specification & block partitioning (QuantumCircuit / IRModule)
//   2. MiddleEnd: Hardware-aware transpilation, gate fusion, and NUMA optimization (Middleware)
//   3. BackEnd: Multi-threaded / distributed execution engine & state vector simulation (ExecutionEngine)
//
// Circuit Execution Workflow:
//   1. Block Validation & Ingestion:
//      Accepts a single QuantumCircuit or a sequential vector of circuit blocks.
//   2. Zero-Copy IR Extraction:
//      Releases the IRModule from the QuantumCircuit via move semantics (zero memory copying).
//   3. MiddleEnd Transpilation:
//      Invokes Middleware::transpile() on the IRModule to apply backend-specific optimizations
//      (e.g., NUMA stride analysis, global SWAP scheduling, gate fusion into UnitaryPool).
//   4. BackEnd Engine Initialization:
//      - Block 0: Allocates and NUMA first-touch initializes the CPUStateVector to |0...0>.
//      - Subsequent Blocks: Transfers the physical CPUStateVector from the previous block to the
//        next block's engine using move semantics (zero-copy state vector continuity).
//   5. Kernel Dispatch & State Mutation:
//      Dispatches gate instructions linearly to the backend execution engine.

#include <cstdint>
#include <complex>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <frontend/frontend.hpp>
#include <middleware/middleware.hpp>
#include <backend/execution_engine.hpp>
#include <backend/cpu_openmp_engine.hpp>
#include <backend/cpu_state_vector.hpp>

namespace qlassical {

    // -------------------------------------------------------------
    // CircuitRunner — High-Level Quantum Circuit Execution Wrapper
    // -------------------------------------------------------------

    class CircuitRunner {
    public:
        // ------------
        // Constructors
        // ------------

        // Constructor for a single QuantumCircuit block.
        // Parameters:
        //   circuit: Shared pointer to the QuantumCircuit to run.
        //   backend: Target backend execution architecture (default: CPU_OPENMP).
        explicit CircuitRunner(std::shared_ptr<QuantumCircuit> circuit,
                               Backend backend = Backend::CPU_OPENMP,
                               middleware::TranspilerConfig config = {})
            : backend_type_(backend)
            , middleware_()
            , transpiler_config_(config)
        {
            if (!circuit) {
                throw std::invalid_argument("CircuitRunner: circuit shared pointer cannot be null");
            }
            circuits_.push_back(std::move(circuit));
        }

        // Constructor for a sequence of QuantumCircuit blocks.
        // Parameters:
        //   circuits: Vector of shared pointers to QuantumCircuit blocks executed sequentially.
        //   backend:  Target backend execution architecture (default: CPU_OPENMP).
        explicit CircuitRunner(std::vector<std::shared_ptr<QuantumCircuit>> circuits,
                               Backend backend = Backend::CPU_OPENMP,
                               middleware::TranspilerConfig config = {})
            : circuits_(std::move(circuits))
            , backend_type_(backend)
            , middleware_()
            , transpiler_config_(config)
        {
            if (circuits_.empty()) {
                throw std::invalid_argument("CircuitRunner: circuit block sequence cannot be empty");
            }
            for (std::size_t i = 0; i < circuits_.size(); ++i) {
                if (!circuits_[i]) {
                    throw std::invalid_argument(
                        "CircuitRunner: null circuit pointer found at block index " +
                        std::to_string(i));
                }
            }
        }

        // Move-only semantics (ExecutionEngine is non-copyable)
        CircuitRunner(CircuitRunner&&) noexcept = default;
        CircuitRunner& operator=(CircuitRunner&&) noexcept = default;
        CircuitRunner(const CircuitRunner&) = delete;
        CircuitRunner& operator=(const CircuitRunner&) = delete;

        ~CircuitRunner() = default;

        // ------------------
        // Execution Workflow
        // ------------------

        // Executes the full quantum circuit pipeline.
        // For multi-block circuits, the StateVector resulting from block (k) is moved
        // with zero-copy semantics to initialize block (k+1).
        void run() {
            if (circuits_.empty()) {
                throw std::runtime_error("CircuitRunner::run: no circuits provided to execute");
            }

            // Preserved physical state vector across consecutive circuit blocks
            backend::CPUStateVector previous_state;
            bool is_first_block = true;

            for (std::size_t b = 0; b < circuits_.size(); ++b) {
                auto& circuit = circuits_[b];
                if (!circuit) {
                    throw std::invalid_argument(
                        "CircuitRunner::run: circuit pointer at block index " +
                        std::to_string(b) + " is null");
                }

                // -------------------------------------------------------------
                // 1. Zero-Copy IR Extraction: Release IRModule from builder
                // -------------------------------------------------------------
                IRModule module = std::move(*circuit).release();

                // -------------------------------------------------------------
                // 2. MiddleEnd Transpilation: Target-aware optimization passes
                // -------------------------------------------------------------
                middleware_.transpile(module, backend_type_, transpiler_config_);

                const uint32_t nq = module.num_qubits;

                // -------------------------------------------------------------
                // 3. BackEnd Engine Setup & State Vector Continuity
                // -------------------------------------------------------------
                switch (backend_type_) {
                    case Backend::CPU_OPENMP: {
                        std::unique_ptr<backend::CPUOpenMPEngine> engine;

                        if (is_first_block) {
                            // First block: Allocate and NUMA first-touch initialize state vector to |0...0>
                            engine = std::make_unique<backend::CPUOpenMPEngine>(nq);
                            is_first_block = false;
                        } else {
                            // Subsequent blocks: Check qubit dimension consistency
                            if (previous_state.num_qubits() != nq) {
                                throw std::invalid_argument(
                                    "CircuitRunner::run: qubit count mismatch between block " +
                                    std::to_string(b - 1) + " (" +
                                    std::to_string(previous_state.num_qubits()) + " qubits) and block " +
                                    std::to_string(b) + " (" + std::to_string(nq) + " qubits)");
                            }
                            // Move state vector from previous block into new engine (zero memory copy)
                            engine = std::make_unique<backend::CPUOpenMPEngine>(std::move(previous_state));
                        }

                        // -------------------------------------------------------------
                        // 4. BackEnd Execution: Linear dispatch of GateInstr stream
                        // -------------------------------------------------------------
                        engine->execute(module.program(), module.unitaries());

                        // -------------------------------------------------------------
                        // 5. State Vector Propagation / Final Retention
                        // -------------------------------------------------------------
                        if (b + 1 == circuits_.size()) {
                            // Final block: Store the execution engine for output inspection
                            engine_ = std::move(engine);
                        } else {
                            // Intermediate block: Move state vector out for the next block
                            previous_state = std::move(*engine).extract_state_vector();
                        }
                        break;
                    }

                    case Backend::MPI:
                        throw std::runtime_error("CircuitRunner::run: MPI backend is not yet implemented");

                    case Backend::GPU:
                        throw std::runtime_error("CircuitRunner::run: GPU backend is not yet implemented");

                    default:
                        throw std::invalid_argument("CircuitRunner::run: unknown backend type specified");
                }
            }
        }

        // ----------------
        // State Inspection
        // ----------------

        // View the final state vector amplitudes (read-only span).
        [[nodiscard]] std::span<const std::complex<double>> state() const {
            if (!engine_) {
                throw std::runtime_error(
                    "CircuitRunner::state: cannot inspect state before calling run()");
            }
            return engine_->state();
        }

        // Total number of qubits in the simulated system.
        [[nodiscard]] uint32_t num_qubits() const {
            if (!engine_) {
                throw std::runtime_error(
                    "CircuitRunner::num_qubits: cannot inspect qubit count before calling run()");
            }
            return engine_->num_qubits();
        }

        // Direct access to the underlying backend ExecutionEngine.
        [[nodiscard]] const backend::ExecutionEngine* engine() const noexcept {
            return engine_.get();
        }

        // Access the configured Middleware instance.
        [[nodiscard]] const Middleware& middleware() const noexcept {
            return middleware_;
        }

        [[nodiscard]] Middleware& middleware() noexcept {
            return middleware_;
        }

        // Selected backend architecture.
        [[nodiscard]] Backend backend() const noexcept {
            return backend_type_;
        }

        // Number of circuit blocks registered in the runner.
        [[nodiscard]] std::size_t num_blocks() const noexcept {
            return circuits_.size();
        }

        // Read-only access to circuit block sequence.
        [[nodiscard]] const std::vector<std::shared_ptr<QuantumCircuit>>& circuits() const noexcept {
            return circuits_;
        }

    private:
        std::vector<std::shared_ptr<QuantumCircuit>> circuits_;
        Backend                                      backend_type_ = Backend::CPU_OPENMP;
        Middleware                                   middleware_;
        middleware::TranspilerConfig                 transpiler_config_;
        std::unique_ptr<backend::ExecutionEngine>   engine_;
    };

} // namespace qlassical

namespace qlassical::frontend {
    // Alias inside frontend namespace for consistency
    using qlassical::CircuitRunner;
} // namespace qlassical::frontend

#endif // QLASSICAL_FRONTEND_CIRCUIT_RUNNER_HPP
