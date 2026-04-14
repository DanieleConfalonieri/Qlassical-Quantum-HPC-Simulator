#ifndef QLASSICAL_BACKEND_CPU_OPENMP_ENGINE_HPP
#define QLASSICAL_BACKEND_CPU_OPENMP_ENGINE_HPP

// -------------------------------------------
// CPUOpenMPEngine — Shared-Memory CPU Backend
// -------------------------------------------
//
// Concrete ExecutionEngine for multi-core CPU execution with OpenMP.


// Architecture:
//   CPUOpenMPEngine owns a CPUStateVector and implements the dispatch loop that
//   converts the Frontend's GateInstr stream into CPUStateVector mutations.


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
//     sub-blocks of the CPUStateVector.                                   
//                                                                       
//     For a k-qubit unitary acting on qubits {q0, ..., q_{k-1}}:      
//       - Block size = 2^k                                              
//       - Number of blocks = 2^n / 2^k = 2^(n-k)                      
//       - Each block is an independent dense mat-vec: U × |block> 
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
#include "state_vector_utils.hpp"
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
                        // Unrecognized Instruction -> thow exception

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

        // Direct CPUStateVector access (testing/debugging)

        [[nodiscard]] const CPUStateVector& state_vector() const noexcept {
            return sv_;
        }

    private:
        CPUStateVector sv_;

        // -----------------------------
        // Algorithm-Driven Gate Kernels 
        // -----------------------------

        //  We don't need to compute the action of the gate as 
        //  a rotation on the Bloch Sphere!

        //  For a single-qubit gate on target qubit t:

        //    stride = 1 << t
        //    for each pair (i, i + stride) where target bit t of i is 0:
        //        a0 = amplitudes[i]
        //        a1 = amplitudes[i + stride]
        //        amplitudes[i]          = U[0][0]*a0 + U[0][1]*a1
        //        amplitudes[i + stride] = U[1][0]*a0 + U[1][1]*a1
        
        //  The outer loop can be parallelized with OpenMP. For low-index
        //  qubits, the stride is small and pairs are cache-local.

        //  For high-index qubits, the stride is large...NUMA-aware page
        //  placement becomes critical for data locality.
        

        // Each kernel uses OpenMP; we generalize the pattern for all gates
        template <typename KernelFunc>
        inline void flat_loop(std::size_t iters, KernelFunc&& kernel) {
 #ifdef QLASSICAL_HAS_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < static_cast<int64_t>(iters); ++i) {
                std::forward<KernelFunc>(kernel)(static_cast<std::size_t>(i)); // Perfect forwarding of the kernel function
            }
        }

        // Hadamard: H = (1/sqrt(2)) [[1, 1], [1, -1]]
        void apply_h(int16_t target) {
            const std::size_t half_dim = sv_.dimension()/2;
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            auto* amp = sv_.data();

            flat_loop(half_dim, [amp, target, inv_sqrt2](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                // Extract amplitudes
                const auto a0 = amp[i0];
                const auto a1 = amp[i1];
                // Apply Hadamard transformation
                amp[i0] = inv_sqrt2 * (a0 + a1);
                amp[i1] = inv_sqrt2 * (a0 - a1);
            });
        }

        // Pauli-X: X = [[0, 1], [1, 0]]
        void apply_x(int16_t target) { 
            // swap amplitude pairs
            // amp[i] <-> amp[i + stride]
            const std::size_t half_dim = sv_.dimension()/2;
            auto* amp = sv_.data();

            flat_loop(half_dim, [amp, target](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                std::swap(amp[i0], amp[i1]);
            });
        }

        // Pauli-Y: Y = [[0, -i], [i, 0]]
        void apply_y(int16_t target) {
            // TODO: implement -> swap with phase
            // amp[i]          = -i * amp[i + stride]
            // amp[i + stride] =  i * amp[i]
            (void)target;
        }

        // Pauli-Z: Z = [[1, 0], [0, -1]] 
        void apply_z(int16_t target) {
            // TODO: implement -> negate amp[i + stride]
            // Only the |1⟩ component gets a sign flip
            (void)target;
        }

        // S gate: S = [[1, 0], [0, i]]
        void apply_s(int16_t target) {
            // TODO: implement -> multiply |1> component by i
            (void)target;
        }

        // RX(theta): [[cos(theta/2), -i sin(theta/2)], [-i sin(theta/2), cos(theta/2)]] 
        void apply_rx(int16_t target, float theta) {
            // TODO: implement -> parametric rotation around X axis
            (void)target;
            (void)theta;
        }

        // RY(theta): [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]] 
        void apply_ry(int16_t target, float theta) {
            // TODO: implement -> parametric rotation around Y axis
            (void)target;
            (void)theta;
        }

        // RZ(theta): [[e^{-i*theta/2}, 0], [0, e^{i*theta/2}]] 
        void apply_rz(int16_t target, float theta) {
            // TODO: implement -> parametric rotation around Z axis
            (void)target;
            (void)theta;
        }

        // CNOT: if control is |1⟩, flip target
        void apply_cx(int16_t control, int16_t target) {
            // TODO: implement -> controlled-X via bit masking
            //
            // Iterate over amplitude pairs that differ in bit 'target'.
            // For each pair, check if bit 'control' is set in the index.
            // If yes, swap the pair (apply X on target). If no, skip.
            //
            // const std::size_t ctrl_mask = std::size_t{1} << control;
            // const std::size_t tgt_stride = std::size_t{1} << target;
            // ...
            // if (i & ctrl_mask) { std::swap(amp[i], amp[i + tgt_stride]); }
            (void)control;
            (void)target;
        }

        // SWAP: exchange amplitudes of two qubits
        void apply_swap(int16_t q0, int16_t q1) {
            // TODO: implement -> swap amplitudes where q0 and q1 differ
            // Decomposable as 3 CNOTs, but direct implementation is faster.
            (void)q0;
            (void)q1;
        }

        // Toffoli (CCX): if both controls are |1>, flip target 
        void apply_ccx(int16_t c0, int16_t c1, int16_t target) {
            // TODO: implement -> doubly-controlled X
            // Same bit-weaving as CX but with two control masks:
            // if ((i & c0_mask) && (i & c1_mask)) { swap pair }
            (void)c0;
            (void)c1;
            (void)target;
        }

        // Fredkin (CSWAP): if control is |1>, swap two targets
        void apply_cswap(int16_t ctrl, int16_t q0, int16_t q1) {
            // TODO: implement -> controlled swap
            (void)ctrl;
            (void)q0;
            (void)q1;
        }

        // -----------------------
        // Data-Driven Gate Kernel 
        // -----------------------
        //
        //  For UNITARY and FUSED_BLOCK gates, the matrix is runtime data
        //  from the UnitaryPool. The idea is:
        //
        //    1. Retrieve the dense matrix: pool.get(instr.matrix_idx)
        //    2. Determine the target qubits: instr.qubits[0..arity-1]
        //    3. For each sub-block of 2^k amplitudes (k = arity):
        //       a. Gather the 2^k amplitudes into a local buffer
        //       b. Apply U * buffer
        //       c. Scatter the results back
        //
        //  This has to be done within the Eigen framework

        void apply_unitary(const GateInstr& instr, const UnitaryPool& pool) {
            // TODO: implement generic unitary application via Eigen::Map
            (void)instr;
            (void)pool;
        }

        // -----------
        // Measurement 
        // -----------
        //
        //  Measurement in an exact CPUStateVector simulator requires:
        //    1. Compute marginal probability with Born's rule
        //    2. Generate random number r
        //    3. Collapse: if r < P(|1>), project onto |1> subspace;
        //       otherwise project onto |0> subspace
        //    4. Renormalize
    

        void apply_measure([[maybe_unused]] int16_t target) {
            // TODO: projection and renormalization
        }
    };

    // ExecutionEngine convenience overload
    
    inline void ExecutionEngine::execute(const IRModule& module) {
        execute(module.program(), module.unitaries());
    }

}

#endif 
