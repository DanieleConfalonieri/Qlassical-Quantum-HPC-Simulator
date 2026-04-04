#ifndef QLASSICAL_BACKEND_CPU_OPENMP_ENGINE_HPP
#define QLASSICAL_BACKEND_CPU_OPENMP_ENGINE_HPP

// -------------------------------------------
// CPUOpenMPEngine - Shared-Memory CPU Backend
// -------------------------------------------
//
// Concrete ExecutionEngine for multi-core CPU execution with OpenMP.


// Architecture:
//   CPUOpenMPEngine owns a StateVector and implements the dispatch loop that
//   converts the Frontend's GateInstr stream into statevector mutations.


// Gate dispatch strategy:
//     - Algorithm-Driven Gates (H, X, Y, Z, S, RX, RY, RZ, CX, SWAP,    
//       CCX, CSWAP)                                                      
//                                                                       
//       The unitary matrix is known at compile time! We write hardcoded   
//       kernels. Given target qubit t, we iterate over  
//       pairs of amplitudes whose indices differ only in bit t, and apply 
//       the 2x2 transformation directly.                                  
//                                                                       
//       For controlled gates (CX, CCX), we additionally mask on the      
//       control bit(s) and skip pairs where the control is |0⟩.          

//     - Data-Driven Gates (UNITARY, FUSED_BLOCK)                         
//                                                                       
//     The unitary matrix is runtime data stored in the UnitaryPool.    
//     We read the matrix via pool.get(matrix_idx), map it with         
//     Eigen::Map, and apply it as a dense matrix-vector product over   
//     sub-blocks of the statevector.                                   
//                                                                       
//     For a k-qubit unitary acting on qubits {q0, ..., q_{k-1}}:      
//       - Block size = 2^k                                              
//       - Number of blocks = 2^n / 2^k = 2^(n-k)                      
//       - Each block is an independent dense mat-vec: U × |block⟩     
//                                                                       
//     This is the fallback path. It's correct for any unitary but      
//     slower than algorithm-driven 


//   Thead model:
//   Thread count is determined entirely by OMP_NUM_THREADS or the runtime
//   default (omp_get_max_threads())


#include <cstdint>
#include <cmath>
#include <complex>
#include <span>
#include <stdexcept>

#include "execution_engine.hpp"
#include "cpu_state_vector.hpp"
#include <frontend/frontend.hpp>

namespace qlassical::backend {

    class CPUOpenMPEngine final : public ExecutionEngine {
    public:
        // Construction:
        // Allocates and initializes a CPUStateVector for num_qubits qubits.
        // Throws std::bad_alloc if the system cannot satisfy 2^n × 16 bytes! 
        explicit CPUOpenMPEngine(uint32_t num_qubits) {
            sv_.initialize(num_qubits);
        }

        // Default move, deleted copy (inherited from ExecutionEngine + CPUStateVector).

        // ExecutionEngine interface 

        void execute(std::span<const GateInstr> program,
                     const UnitaryPool& pool) override {
            for (const auto& instr : program) {
                switch (instr.type) {

                    // -----------------------
                    //  Algorithm-Driven Gates 
                    // -----------------------

                    // Single-qubit (arity 1)
                    case GateType::H:
                        apply_h(instr.qubits[0]);
                        break;
                    case GateType::X:
                        apply_x(instr.qubits[0]);
                        break;
                    case GateType::Y:
                        apply_y(instr.qubits[0]);
                        break;
                    case GateType::Z:
                        apply_z(instr.qubits[0]);
                        break;
                    case GateType::S:
                        apply_s(instr.qubits[0]);
                        break;

                    // Single-qubit parametric (arity 1, params > 0)
                    case GateType::RX:
                        apply_rx(instr.qubits[0], instr.params[0]);
                        break;
                    case GateType::RY:
                        apply_ry(instr.qubits[0], instr.params[0]);
                        break;
                    case GateType::RZ:
                        apply_rz(instr.qubits[0], instr.params[0]);
                        break;

                    // Two-qubit (arity 2)
                    case GateType::CX:
                        apply_cx(instr.qubits[0], instr.qubits[1]);
                        break;
                    case GateType::SWAP:
                        apply_swap(instr.qubits[0], instr.qubits[1]);
                        break;

                    // Three-qubit (arity 3)
                    case GateType::CCX:
                        apply_ccx(instr.qubits[0], instr.qubits[1],
                                  instr.qubits[2]);
                        break;
                    case GateType::CSWAP:
                        apply_cswap(instr.qubits[0], instr.qubits[1],
                                    instr.qubits[2]);
                        break;

                    // -----------------
                    // Data-Driven Gates
                    // -----------------
                    case GateType::UNITARY:
                        apply_unitary(instr, pool);
                        break; 
                    case GateType::FUSED_BLOCK:
                        apply_unitary(instr, pool);
                        break;

                    // ----------
                    // Directives
                    // ----------

                    case GateType::BARRIER:
                        // Scheduling hint for the Middle-End. No-op at execution.
                        break;

                    case GateType::MEASURE:
                        apply_measure(instr.qubits[0]);
                        break;

                    case GateType::GLOBAL_SWAP:
                        // TODO: physical qubit permutation (streaming memory sweep)
                        break;
                    default:
                        throw std::runtime_error(
                            "CPUOpenMPEngine::execute: unrecognized gate type " +
                            std::to_string(static_cast<uint8_t>(instr.type)));
                        
                }
            }
        }
    
    [[nodiscard]] std::span<const std::complex<double>>
        state() const override {
            return sv_.amplitudes();
        }

        [[nodiscard]] uint32_t num_qubits() const override {
            return sv_.num_qubits();
        }

        // Direct StateVector access (testing/debugging)

        [[nodiscard]] const CPUStateVector& state_vector() const noexcept {
            return sv_;
        }
    
    private:
        CPUStateVector sv_;
    };
} 
#endif 