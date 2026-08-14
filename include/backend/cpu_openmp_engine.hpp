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
                        apply_rx(instr.qubits[0], instr.payload.param);
                        break;
                    case GateType::RY:
                        apply_ry(instr.qubits[0], instr.payload.param);
                        break;
                    case GateType::RZ:
                        apply_rz(instr.qubits[0], instr.payload.param);
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

        [[nodiscard]] std::span<const std::complex<double>> state() const override {
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
        

        // We generalize the pattern for all single-qubit gates.
        template <typename KernelFunc>
        inline void single_target_loop(std::size_t iters, KernelFunc&& kernel) {
            
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(iters); ++i) {
                std::forward<KernelFunc>(kernel)(static_cast<std::size_t>(i)); // Perfect forwarding of the kernel function
            }
        }

        // We generalize the pattern for two-qubit gates.
        template <typename KernelFunc>
        inline void double_hole_loop(int16_t q0, int16_t q1, KernelFunc&& kernel) {
            // A 2-qubit gate acts on half of the state vector with respect to 1-qubit gates.
            const std::size_t iters = sv_.dimension() / 4;
            
            const int16_t q_min = std::min(q0, q1);
            const int16_t q_max = std::max(q0, q1);

            // We implement the same bit-level trick as 1q gates, but now we have two "holes" in the index.
            // We split the index into three parts: low, mid and high.
            const std::size_t mask_low = (std::size_t{1} << q_min) - 1;
            const std::size_t mask_mid = (std::size_t{1} << (q_max - q_min - 1)) - 1;

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(iters); ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);
                
                // 1. Extract the three parts.
                const std::size_t low  = idx & mask_low;
                const std::size_t mid  = (idx >> q_min) & mask_mid;
                const std::size_t high = idx >> (q_max - 1);

                // 2. Reassemble with two holes (q0 and q1 are 0 by construction).
                std::size_t base_idx = low | (mid << (q_min + 1)) | (high << (q_max + 1));

                // 3. Call the kernel with the base index.
                std::forward<KernelFunc>(kernel)(base_idx); 
            }
        }

        // We generalize the pattern for three-qubit gates.
        template <typename KernelFunc>
        inline void triple_hole_loop(int16_t q0, int16_t q1, int16_t q2, KernelFunc&& kernel) {
            // A 3-qubit gate isolates a subspace of dimension N/8.
            const std::size_t iters = sv_.dimension() / 8;
            
            
            const int16_t q_min = std::min({q0, q1, q2});
            const int16_t q_max = std::max({q0, q1, q2});
            const int16_t q_mid = q0 + q1 + q2 - q_min - q_max; // trick to find the middle qubit without sorting

            // Here we define three masks, having 4 parts.
            const std::size_t mask_low  = (std::size_t{1} << q_min) - 1;
            const std::size_t mask_mid1 = (std::size_t{1} << (q_mid - q_min - 1)) - 1;
            const std::size_t mask_mid2 = (std::size_t{1} << (q_max - q_mid - 1)) - 1;

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(iters); ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);
                
                // 1. Extract the four parts.
                const std::size_t low  = idx & mask_low;
                const std::size_t mid1 = (idx >> q_min) & mask_mid1;
                const std::size_t mid2 = (idx >> (q_mid - 1)) & mask_mid2;
                const std::size_t high = idx >> (q_max - 2);

                // 2. Reassemble with three holes(q_min, q_mid, q_max are 0 by construction).
                const std::size_t base_idx = low 
                                           | (mid1 << (q_min + 1)) 
                                           | (mid2 << (q_mid + 1)) 
                                           | (high << (q_max + 1));

                // 3. Call the kernel with the base index.
                std::forward<KernelFunc>(kernel)(base_idx);
            }
        }

        // Hadamard: H = (1/sqrt(2)) [[1, 1], [1, -1]]
        void apply_h(int16_t target) {
            const std::size_t half_dim = sv_.dimension()/2;
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target, inv_sqrt2](std::size_t idx) {
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

            single_target_loop(half_dim, [amp, target](std::size_t idx) {
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
            const std::size_t half_dim = sv_.dimension()/2;
            auto* amp = sv_.data();
            single_target_loop(half_dim, [amp, target](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                const auto a0 = amp[i0];
                const auto a1 = amp[i1];
                // we avoid creating a temporary std::complex<double> for i; it is equivalent to swap real and imaginary parts with a sign change
                // -i * a1  -> -i * (re1 + i*im1) = im1 - i*re1
                amp[i0] = std::complex<double>(a1.imag(), -a1.real());
                //  i * a0  ->  i * (re0 + i*im0) = -im0 + i*re0
                amp[i1] = std::complex<double>(-a0.imag(), a0.real());
            });
        }

        // Pauli-Z: Z = [[1, 0], [0, -1]] 
        void apply_z(int16_t target) {
            // TODO: implement -> negate amp[i + stride]
            // Only the |1⟩ component gets a sign flip
            const std::size_t half_dim = sv_.dimension()/2;
            auto* amp = sv_.data();
            single_target_loop(half_dim, [amp, target](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                amp[i1] = -amp[i1]; // Negate the |1⟩ component
            });
        }

        // S gate: S = [[1, 0], [0, i]]
        void apply_s(int16_t target) {
            // TODO: implement -> multiply |1> component by i
            const std::size_t half_dim = sv_.dimension()/2;
            auto* amp = sv_.data();
            single_target_loop(half_dim, [amp, target](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                // again, avoid creating a temporary std::complex<double> for i; it is equivalent to swap real and imaginary parts with a sign change
                const double re = amp[i1].real();
                const double im = amp[i1].imag();
                amp[i1] = std::complex<double>(-im, re);
            });
        }

        // RX(theta): [[cos(theta/2), -i sin(theta/2)], [-i sin(theta/2), cos(theta/2)]] 
        void apply_rx(int16_t target, float theta) {
            const double half_theta = static_cast<double>(theta) * 0.5;
            // Precompute cos and sin for efficiency
            const double cos = std::cos(half_theta);
            const double sin = std::sin(half_theta);
            
            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target, cos, sin](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                
                const auto a0 = amp[i0];
                const auto a1 = amp[i1];
                
                // Usual multiplication trick
                // -i * s * a1 = s * (im1 - i*re1)
                amp[i0] = cos * a0 + std::complex<double>(sin * a1.imag(), -sin * a1.real());
                amp[i1] = std::complex<double>(sin * a0.imag(), -sin * a0.real()) + cos * a1;
            });
        }

        // RY(theta): [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]] 
        void apply_ry(int16_t target, float theta) {
            const double half_theta = static_cast<double>(theta) * 0.5;
            const double cos = std::cos(half_theta);
            const double sin = std::sin(half_theta);
            
            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target, cos, sin](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                
                const auto a0 = amp[i0];
                const auto a1 = amp[i1];
                
                amp[i0] = cos * a0 - sin * a1;
                amp[i1] = sin * a0 + cos * a1;
            });
        }

        // RZ(theta): [[e^{-i*theta/2}, 0], [0, e^{i*theta/2}]] 
        void apply_rz(int16_t target, float theta) {
            const double half_theta = static_cast<double>(theta) * 0.5;
            // Precompute the complex exponentials for efficiency   
            const std::complex<double> p0(std::cos(-half_theta), std::sin(-half_theta));
            const std::complex<double> p1(std::cos(half_theta), std::sin(half_theta));
            
            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target, p0, p1](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                
                amp[i0] *= p0;
                amp[i1] *= p1;
            });
        }

        // CNOT: if control is |1⟩, flip target
        void apply_cx(int16_t control, int16_t target) {
            // TODO: implement -> controlled-X via bit masking
            //
            // Iterate over amplitude pairs that differ in bit 'target'.
            // For each pair, checking if bit 'control' is set in the index, would lead to divergence 
            // and 50% of the times this would lead to a no-op.
            // Instead, we define the custom logic to iterate only over the indices where control is |1⟩ in the controlled_2q_loop. 
            // We just need to  swap the pair (apply X on target). 
            auto* amp = sv_.data();
            const std::size_t ctrl_mask = std::size_t{1} << control;
            const std::size_t tgt_mask  = std::size_t{1} << target;

            double_hole_loop(control, target, [amp, ctrl_mask, tgt_mask](std::size_t base_idx) {
                const std::size_t i0 = base_idx | ctrl_mask; // control is |1⟩, target is |0⟩
                const std::size_t i1 = base_idx | ctrl_mask | tgt_mask; // control is |1⟩, target is |1⟩

                std::swap(amp[i0], amp[i1]); 
            });
        }

        // SWAP: exchange amplitudes of two qubits
        // It works only on the pairs of amplitudes where the two qubits differ (|01⟩ and |10⟩).
        void apply_swap(int16_t q0, int16_t q1) {
            auto* amp = sv_.data();
            const std::size_t mask0 = std::size_t{1} << q0;
            const std::size_t mask1 = std::size_t{1} << q1;

            double_hole_loop(q0, q1, [amp, mask0, mask1](std::size_t base_idx) {
                // Construct the two indices |01⟩ and |10⟩
                const std::size_t i01 = base_idx | mask1; // q0 is 0, q1 is 1
                const std::size_t i10 = base_idx | mask0; // q0 is 1, q1 is 0

                std::swap(amp[i01], amp[i10]);
            });
        }

        // Toffoli (CCX): if both controls are |1>, flip target 
        void apply_ccx(int16_t c0, int16_t c1, int16_t target) {
            // TODO: implement -> doubly-controlled X
            // We use the optimized triple_hole_loop strategy again to avoid
            // branching and inefficiencies.
            auto* amp = sv_.data();
            const std::size_t ctrl_mask0 = std::size_t{1} << c0;
            const std::size_t ctrl_mask1 = std::size_t{1} << c1;
            const std::size_t tgt_mask   = std::size_t{1} << target;

            const std::size_t ctrls_mask = ctrl_mask0 | ctrl_mask1;

            triple_hole_loop(c0, c1, target, [amp, ctrls_mask, tgt_mask](std::size_t base_idx) {
                // Building the index for c0=1, c1=1, target=0
                const std::size_t i0 = base_idx | ctrls_mask;
                // Building the index for c0=1, c1=1, target=1
                const std::size_t i1 = i0 | tgt_mask;
                
                std::swap(amp[i0], amp[i1]);
            });
        }

        // Fredkin (CSWAP): if control is |1>, swap two targets
        void apply_cswap(int16_t ctrl, int16_t q0, int16_t q1) {
            // TODO: implement -> controlled swap
            auto* amp = sv_.data();
            const std::size_t ctrl_mask = std::size_t{1} << ctrl;
            const std::size_t tgt_mask0   = std::size_t{1} << q0;
            const std::size_t tgt_mask1   = std::size_t{1} << q1;

            triple_hole_loop(ctrl, q0, q1, [amp, ctrl_mask, tgt_mask0, tgt_mask1](std::size_t base_idx) {
                // Building the index for ctrl=1, q0=0, q1=1
                const std::size_t i01 = base_idx | ctrl_mask | tgt_mask1; // control is |1⟩, q0 is |0⟩, q1 is |1⟩
                const std::size_t i10 = base_idx | ctrl_mask | tgt_mask0; // control is |1⟩, q0 is |1⟩, q1 is |0⟩

                std::swap(amp[i01], amp[i10]);
            });
        }

        // -----------------------
        // Data-Driven Gate Kernel 
        // -----------------------
        //
        //  For UNITARY and FUSED_BLOCK gates, the matrix is runtime data
        //  from the UnitaryPool. The idea is:
        //
        //    1. Retrieve the dense matrix: pool.get(instr.payload.matrix_idx)
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
