#ifndef QLASSICAL_BACKEND_CPU_OPENMP_ENGINE_HPP
#define QLASSICAL_BACKEND_CPU_OPENMP_ENGINE_HPP

// -------------------------------------------------------------
// CPUOpenMPEngine — Shared-Memory CPU Execution Engine
// -------------------------------------------------------------
//
// Concrete ExecutionEngine for multi-core CPU execution with OpenMP.
//
// Architecture:
//   CPUOpenMPEngine owns a CPUStateVector and implements the dispatch loop that
//   converts the Frontend's GateInstr stream into CPUStateVector mutations.
//
// Gate Dispatch Strategy:
//   - Algorithm-Driven Gates (H, X, Y, Z, S, RX, RY, RZ, CX, SWAP, CCX, CSWAP):
//     The unitary matrix is known at compile time. Specialized kernels iterate over
//     amplitude pairs differing in target bit t and apply the 2x2 transformation directly.
//     For controlled gates (CX, CCX, CSWAP), bitmasks isolate the control subspace
//     to avoid branching and thread divergence.
//
//   - Data-Driven Gates (UNITARY, FUSED_BLOCK):
//     The unitary matrix is runtime data stored in the UnitaryPool.
//     The engine maps the matrix via Eigen::Map and performs dense matrix-vector
//     multiplication over sub-blocks of the state vector via gather-scatter.
//
//   - Circuit Directives (BARRIER, MEASURE, GLOBAL_SWAP):
//     Directives control circuit scheduling, measurement collapse, and state layout.
//
// Thread Model:
//   Thread count is determined entirely by OMP_NUM_THREADS or omp_get_max_threads().
//   Static scheduling matches NUMA first-touch page placement.

#include <cmath>
#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>
#include <random>
#include <optional>
#include <algorithm>
#include <Eigen/Dense>

#include <frontend/frontend.hpp>
#include "cpu_state_vector.hpp"
#include "execution_engine.hpp"
#include "state_vector_utils.hpp"

namespace qlassical::backend {

    class CPUOpenMPEngine final : public ExecutionEngine {
    public:
        // Bring base class overloads into scope to prevent name hiding
        using ExecutionEngine::execute;

        // ------------
        // Constructors
        // ------------

        // Constructor allocating and initializing state vector for num_qubits qubits.
        // Parameters:
        //   num_qubits:   Number of qubits in the simulation.
        //   seed:         Optional seed for deterministic measurement RNG.
        //   reserve_meas: Optional preallocation count for measurement results.
        explicit CPUOpenMPEngine(uint32_t num_qubits,
                                 std::optional<uint64_t> seed = std::nullopt,
                                 std::size_t reserve_meas = 0) { 
            sv_.initialize(num_qubits); 
            
            if (seed.has_value()) {
                rng_.seed(seed.value());
            } else {
                std::random_device rd;
                rng_.seed(rd());
            }
            
            if (reserve_meas > 0) {
                measurements_.reserve(reserve_meas);
            }
        }

        // Constructor from an initial state vector (must be normalized).
        // Parameters:
        //   initial_state: Initial state vector amplitudes.
        //   seed:          Optional seed for deterministic measurement RNG.
        //   reserve_meas:  Optional preallocation count for measurement results.
        explicit CPUOpenMPEngine(std::span<const std::complex<double>> initial_state,
                                 std::optional<uint64_t> seed = std::nullopt,
                                 std::size_t reserve_meas = 0) {
            sv_.initialize(initial_state);
            
            if (seed.has_value()) {
                rng_.seed(seed.value());
            } else {
                std::random_device rd;
                rng_.seed(rd());
            }
            
            if (reserve_meas > 0) {
                measurements_.reserve(reserve_meas);
            }
        }

        // Constructor transferring ownership of an existing CPUStateVector (zero-copy move).
        explicit CPUOpenMPEngine(CPUStateVector&& sv) noexcept
            : sv_(std::move(sv))
        {}

        // Transfer / assign an existing CPUStateVector (zero-copy move).
        void set_state_vector(CPUStateVector&& sv) noexcept {
            sv_ = std::move(sv);
        }

        // -------------------------
        // ExecutionEngine Interface
        // -------------------------

        // Reset state vector to |0...0> and clear measurements.
        void reset() override {
            sv_.initialize(sv_.num_qubits());
            measurements_.clear();
        }

        // Reset to an arbitrary state vector (must be normalized).
        void reset(std::span<const std::complex<double>> initial_state) override {
            sv_.initialize(initial_state);
            measurements_.clear();
        }

        void set_seed(uint64_t seed) override {
            rng_.seed(seed);
        }

        void reserve_measurements(std::size_t count) override {
            measurements_.reserve(count);
        }

        [[nodiscard]] const std::vector<uint8_t>& measurements() const override {
            return measurements_;
        }

        // Dispatches the linear gate instruction stream.
        void execute(std::span<const GateInstr> program,
                     const UnitaryPool& pool) override {
            for (const auto& instr : program) {
                switch (instr.type) {

                // -----------------------
                // Algorithm-Driven Gates
                // -----------------------

                // Single-qubit gates (arity 1)
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

                // Single-qubit parametric gates (arity 1, params > 0)
                case GateType::RX:
                    apply_rx(instr.qubits[0], instr.payload.param);
                    break;
                case GateType::RY:
                    apply_ry(instr.qubits[0], instr.payload.param);
                    break;
                case GateType::RZ:
                    apply_rz(instr.qubits[0], instr.payload.param);
                    break;

                // Two-qubit gates (arity 2)
                case GateType::CX:
                    apply_cx(instr.qubits[0], instr.qubits[1]);
                    break;
                case GateType::SWAP:
                    apply_swap(instr.qubits[0], instr.qubits[1]);
                    break;

                // Three-qubit gates (arity 3)
                case GateType::CCX:
                    apply_ccx(instr.qubits[0], instr.qubits[1], instr.qubits[2]);
                    break;
                case GateType::CSWAP:
                    apply_cswap(instr.qubits[0], instr.qubits[1], instr.qubits[2]);
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

                // ----------------------------
                // Circuit Structure Directives
                // ----------------------------
                case GateType::BARRIER:
                    // Scheduling hint for compiler passes. No-op at execution.
                    break;

                case GateType::MEASURE:
                    apply_measure(instr.qubits[0]);
                    break;

                case GateType::GLOBAL_SWAP:
                    // Physical qubit permutation (streaming memory sweep)
                    break;

                default:
                    throw std::runtime_error("CPUOpenMPEngine: Unrecognized instruction type");
                }
            }
        }

        // ----------------
        // State Inspection
        // ----------------

        [[nodiscard]] std::span<const std::complex<double>> state() const override {
            return sv_.amplitudes();
        }

        [[nodiscard]] uint32_t num_qubits() const override {
            return sv_.num_qubits();
        }

        // Direct CPUStateVector access (testing/debugging)
        [[nodiscard]] CPUStateVector& state_vector() noexcept {
            return sv_;
        }

        // Extract / transfer ownership of the state vector (zero-copy move).
        [[nodiscard]] CPUStateVector extract_state_vector() && noexcept {
            return std::move(sv_);
        }

    private:
        CPUStateVector                         sv_;
        std::vector<uint8_t>                   measurements_;
        std::mt19937_64                        rng_;
        std::uniform_real_distribution<double> dist_{0.0, 1.0};

        // -----------------------------
        // Algorithm-Driven Gate Kernels
        // -----------------------------

        // General loop pattern for single-qubit gate kernels.
        template <typename KernelFunc>
        inline void single_target_loop(std::size_t iters, KernelFunc&& kernel) {
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(iters); ++i) {
                std::forward<KernelFunc>(kernel)(static_cast<std::size_t>(i));
            }
        }

        // General loop pattern for two-qubit gate kernels.
        // Splits indices to create two zero-bit holes at positions q0 and q1.
        template <typename KernelFunc>
        inline void double_hole_loop(uint8_t q0, uint8_t q1, KernelFunc&& kernel) {
            // A 2-qubit gate acts on half of the state vector dimension compared to 1-qubit gates
            const std::size_t iters = sv_.dimension() / 4;

            const uint8_t q_min = std::min(q0, q1);
            const uint8_t q_max = std::max(q0, q1);

            // Split the index into three parts: low, mid, and high
            const std::size_t mask_low = (std::size_t{1} << q_min) - 1;
            const std::size_t mask_mid = (std::size_t{1} << (q_max - q_min - 1)) - 1;

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(iters); ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);

                // 1. Extract the three parts
                const std::size_t low  = idx & mask_low;
                const std::size_t mid  = (idx >> q_min) & mask_mid;
                const std::size_t high = idx >> (q_max - 1);

                // 2. Reassemble with two zero-bit holes (q_min and q_max are 0 by construction)
                const std::size_t base_idx = low | (mid << (q_min + 1)) | (high << (q_max + 1));

                // 3. Dispatch kernel on base index
                std::forward<KernelFunc>(kernel)(base_idx);
            }
        }

        // General loop pattern for three-qubit gate kernels.
        // Splits indices to create three zero-bit holes at positions q0, q1, and q2.
        template <typename KernelFunc>
        inline void triple_hole_loop(uint8_t q0, uint8_t q1, uint8_t q2,
                                     KernelFunc&& kernel) {
            // A 3-qubit gate isolates a subspace of dimension N/8
            const std::size_t iters = sv_.dimension() / 8;

            const uint8_t q_min = std::min({q0, q1, q2});
            const uint8_t q_max = std::max({q0, q1, q2});
            const uint8_t q_mid = static_cast<uint8_t>(q0 + q1 + q2 - q_min - q_max);

            // Define masks for the 4 index segments
            const std::size_t mask_low  = (std::size_t{1} << q_min) - 1;
            const std::size_t mask_mid1 = (std::size_t{1} << (q_mid - q_min - 1)) - 1;
            const std::size_t mask_mid2 = (std::size_t{1} << (q_max - q_mid - 1)) - 1;

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(iters); ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);

                // 1. Extract the four parts
                const std::size_t low  = idx & mask_low;
                const std::size_t mid1 = (idx >> q_min) & mask_mid1;
                const std::size_t mid2 = (idx >> (q_mid - 1)) & mask_mid2;
                const std::size_t high = idx >> (q_max - 2);

                // 2. Reassemble with three zero-bit holes (q_min, q_mid, q_max are 0 by construction)
                const std::size_t base_idx = low | (mid1 << (q_min + 1)) |
                                             (mid2 << (q_mid + 1)) |
                                             (high << (q_max + 1));

                // 3. Dispatch kernel on base index
                std::forward<KernelFunc>(kernel)(base_idx);
            }
        }

        // Hadamard: H = (1/sqrt(2)) [[1, 1], [1, -1]]
        void apply_h(uint8_t target) {
            const std::size_t half_dim = sv_.dimension() / 2;
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target, inv_sqrt2](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                const auto a0 = amp[i0];
                const auto a1 = amp[i1];
                amp[i0] = inv_sqrt2 * (a0 + a1);
                amp[i1] = inv_sqrt2 * (a0 - a1);
            });
        }

        // Pauli-X: X = [[0, 1], [1, 0]]
        void apply_x(uint8_t target) {
            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                std::swap(amp[i0], amp[i1]);
            });
        }

        // Pauli-Y: Y = [[0, -i], [i, 0]]
        void apply_y(uint8_t target) {
            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                const auto a0 = amp[i0];
                const auto a1 = amp[i1];
                // -i * a1 = im1 - i*re1
                amp[i0] = std::complex<double>(a1.imag(), -a1.real());
                //  i * a0 = -im0 + i*re0
                amp[i1] = std::complex<double>(-a0.imag(), a0.real());
            });
        }

        // Pauli-Z: Z = [[1, 0], [0, -1]]
        void apply_z(uint8_t target) {
            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                amp[i1] = -amp[i1];
            });
        }

        // S gate: S = [[1, 0], [0, i]]
        void apply_s(uint8_t target) {
            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);
                const double re = amp[i1].real();
                const double im = amp[i1].imag();
                amp[i1] = std::complex<double>(-im, re);
            });
        }

        // RX(theta): [[cos(theta/2), -i sin(theta/2)], [-i sin(theta/2), cos(theta/2)]]
        void apply_rx(uint8_t target, float theta) {
            const double half_theta = static_cast<double>(theta) * 0.5;
            const double cos_val = std::cos(half_theta);
            const double sin_val = std::sin(half_theta);

            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target, cos_val, sin_val](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);

                const auto a0 = amp[i0];
                const auto a1 = amp[i1];

                amp[i0] = cos_val * a0 + std::complex<double>(sin_val * a1.imag(), -sin_val * a1.real());
                amp[i1] = std::complex<double>(sin_val * a0.imag(), -sin_val * a0.real()) + cos_val * a1;
            });
        }

        // RY(theta): [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        void apply_ry(uint8_t target, float theta) {
            const double half_theta = static_cast<double>(theta) * 0.5;
            const double cos_val = std::cos(half_theta);
            const double sin_val = std::sin(half_theta);

            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            single_target_loop(half_dim, [amp, target, cos_val, sin_val](std::size_t idx) {
                const std::size_t i0 = qlassical::backend::utils::insert_zero_bit(idx, target);
                const std::size_t i1 = qlassical::backend::utils::flip_target_bit(i0, target);

                const auto a0 = amp[i0];
                const auto a1 = amp[i1];

                amp[i0] = cos_val * a0 - sin_val * a1;
                amp[i1] = sin_val * a0 + cos_val * a1;
            });
        }

        // RZ(theta): [[e^{-i*theta/2}, 0], [0, e^{i*theta/2}]]
        void apply_rz(uint8_t target, float theta) {
            const double half_theta = static_cast<double>(theta) * 0.5;
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

        // CNOT (CX): if control is |1⟩, flip target
        void apply_cx(uint8_t control, uint8_t target) {
            auto* amp = sv_.data();
            const std::size_t ctrl_mask = std::size_t{1} << control;
            const std::size_t tgt_mask = std::size_t{1} << target;

            double_hole_loop(
                control, target, [amp, ctrl_mask, tgt_mask](std::size_t base_idx) {
                    const std::size_t i0 = base_idx | ctrl_mask;            // control is |1⟩, target is |0⟩
                    const std::size_t i1 = base_idx | ctrl_mask | tgt_mask; // control is |1⟩, target is |1⟩

                    std::swap(amp[i0], amp[i1]);
                });
        }

        // SWAP: exchange amplitudes of two qubits
        void apply_swap(uint8_t q0, uint8_t q1) {
            auto* amp = sv_.data();
            const std::size_t mask0 = std::size_t{1} << q0;
            const std::size_t mask1 = std::size_t{1} << q1;

            double_hole_loop(q0, q1, [amp, mask0, mask1](std::size_t base_idx) {
                const std::size_t i01 = base_idx | mask1; // q0 is |0⟩, q1 is |1⟩
                const std::size_t i10 = base_idx | mask0; // q0 is |1⟩, q1 is |0⟩

                std::swap(amp[i01], amp[i10]);
            });
        }

        // Toffoli (CCX): if both controls are |1⟩, flip target
        void apply_ccx(uint8_t c0, uint8_t c1, uint8_t target) {
            auto* amp = sv_.data();
            const std::size_t ctrl_mask0 = std::size_t{1} << c0;
            const std::size_t ctrl_mask1 = std::size_t{1} << c1;
            const std::size_t tgt_mask   = std::size_t{1} << target;
            const std::size_t ctrls_mask = ctrl_mask0 | ctrl_mask1;

            triple_hole_loop(c0, c1, target,
                             [amp, ctrls_mask, tgt_mask](std::size_t base_idx) {
                                 const std::size_t i0 = base_idx | ctrls_mask;            // c0=1, c1=1, target=0
                                 const std::size_t i1 = i0 | tgt_mask;                    // c0=1, c1=1, target=1

                                 std::swap(amp[i0], amp[i1]);
                             });
        }

        // Fredkin (CSWAP): if control is |1⟩, swap two targets
        void apply_cswap(uint8_t ctrl, uint8_t q0, uint8_t q1) {
            auto* amp = sv_.data();
            const std::size_t ctrl_mask = std::size_t{1} << ctrl;
            const std::size_t tgt_mask0 = std::size_t{1} << q0;
            const std::size_t tgt_mask1 = std::size_t{1} << q1;

            triple_hole_loop(
                ctrl, q0, q1,
                [amp, ctrl_mask, tgt_mask0, tgt_mask1](std::size_t base_idx) {
                    const std::size_t i01 = base_idx | ctrl_mask | tgt_mask1; // ctrl=1, q0=0, q1=1
                    const std::size_t i10 = base_idx | ctrl_mask | tgt_mask0; // ctrl=1, q0=1, q1=0

                    std::swap(amp[i01], amp[i10]);
                });
        }

        // -----------------------
        // Data-Driven Gate Kernel
        // -----------------------
        //
        // Applies arbitrary k-qubit unitary (UNITARY, FUSED_BLOCK) using runtime matrix data:
        //   1. Retrieves the dense matrix from the UnitaryPool.
        //   2. Precomputes gather/scatter byte offsets for target qubit configurations.
        //   3. For each sub-block of 2^k amplitudes, gathers into an aligned buffer,
        //      computes U * v via Eigen SIMD matrix-vector multiplication, and scatters back.
        void apply_unitary(const GateInstr& instr, const UnitaryPool& pool) {
            // 1. Extract arity and unitary matrix from pool
            const uint8_t arity = instr.arity();
            const std::size_t block_size = std::size_t{1} << arity; // 2^k

            auto matrix_span = pool.get(instr.payload.matrix_idx);
            using MatrixType =
                Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>>;
            MatrixType U(matrix_span.data(), block_size, block_size);

            // 2. Order target qubits to correctly generate base index
            std::array<uint8_t, 4> sorted_qubits = {0};
            std::copy_n(instr.qubits, arity, sorted_qubits.begin());
            std::sort(sorted_qubits.begin(), sorted_qubits.begin() + arity);

            // 3. Precompute gather/scatter offsets for target qubit configurations
            std::array<std::size_t, 16> scatter_offsets = {0};

            for (std::size_t j = 0; j < block_size; ++j) {
                std::size_t offset = 0;
                for (uint8_t q = 0; q < arity; ++q) {
                    if ((j >> q) & 1) {
                        offset |= (std::size_t{1} << instr.qubits[q]);
                    }
                }
                scatter_offsets[j] = offset;
            }

            const std::size_t iters = sv_.dimension() >> arity;
            auto* amp = sv_.data();

            // 4. Parallel execution loop
            #pragma omp parallel
            {
                // Stack-allocated buffers to prevent dynamic allocations inside thread region
                alignas(32) std::array<std::complex<double>, 16> buf_in;
                alignas(32) std::array<std::complex<double>, 16> buf_out;

                Eigen::Map<Eigen::VectorXcd> v_in(buf_in.data(), block_size);
                Eigen::Map<Eigen::VectorXcd> v_out(buf_out.data(), block_size);

                #pragma omp for schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(iters); ++i) {
                    // Reconstruct base index by inserting zero bits at target qubit positions
                    std::size_t base_idx = static_cast<std::size_t>(i);
                    for (uint8_t q = 0; q < arity; ++q) {
                        const std::size_t low_mask = (std::size_t{1} << sorted_qubits[q]) - 1;
                        const std::size_t low  = base_idx & low_mask;
                        const std::size_t high = (base_idx & ~low_mask) << 1;
                        base_idx = low | high;
                    }

                    // 1. Gather using precomputed offsets
                    for (std::size_t j = 0; j < block_size; ++j) {
                        buf_in[j] = amp[base_idx | scatter_offsets[j]];
                    }

                    // 2. Vectorized matrix-vector multiplication via Eigen
                    v_out.noalias() = U * v_in;

                    // 3. Vectorized scatter
                    for (std::size_t j = 0; j < block_size; ++j) {
                        amp[base_idx | scatter_offsets[j]] = buf_out[j];
                    }
                }
            }
        }

        // -----------
        // Measurement
        // -----------

        // Measures a single qubit in the computational basis:
        //   1. Computes marginal probability P(|1>) using Born's rule.
        //   2. Generates random sample r in [0, 1).
        //   3. Collapses state vector onto the observed subspace (|0> or |1>).
        //   4. Renormalizes remaining non-zero amplitudes.
        void apply_measure(uint8_t target) {
            const std::size_t half_dim = sv_.dimension() / 2;
            auto* amp = sv_.data();

            // 1. Compute marginal probability P(|1>)
            double p1 = 0.0;
            
            #pragma omp parallel for reduction(+:p1) schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(half_dim); ++i) {
                const std::size_t idx1 =
                    qlassical::backend::utils::insert_zero_bit(static_cast<std::size_t>(i), target) |
                    (std::size_t{1} << target);
                p1 += std::norm(amp[idx1]);
            }

            // 2. Generate random sample r in [0, 1)
            double r = dist_(rng_);

            // 3. Determine outcome
            uint8_t outcome = (r < p1) ? 1 : 0;
            measurements_.push_back(outcome);

            // 4. Renormalize and collapse
            double p_outcome = outcome ? p1 : (1.0 - p1);
            
            // Numerical stability clamp
            if (p_outcome < 1e-12) {
                p_outcome = 1e-12;
            }
            
            const double inv_norm = 1.0 / std::sqrt(p_outcome);

            #pragma omp parallel
            {
                if (outcome == 1) {
                    #pragma omp for schedule(static)
                    for (int64_t i = 0; i < static_cast<int64_t>(half_dim); ++i) {
                        const std::size_t idx0 =
                            qlassical::backend::utils::insert_zero_bit(static_cast<std::size_t>(i), target);
                        const std::size_t idx1 = idx0 | (std::size_t{1} << target);
                        amp[idx0] = {0.0, 0.0};
                        amp[idx1] *= inv_norm;
                    }
                } else {
                    #pragma omp for schedule(static)
                    for (int64_t i = 0; i < static_cast<int64_t>(half_dim); ++i) {
                        const std::size_t idx0 =
                            qlassical::backend::utils::insert_zero_bit(static_cast<std::size_t>(i), target);
                        const std::size_t idx1 = idx0 | (std::size_t{1} << target);
                        amp[idx0] *= inv_norm;
                        amp[idx1] = {0.0, 0.0};
                    }
                }
            }
        }
    };

    // ------------------------------------
    // ExecutionEngine Convenience Overload
    // ------------------------------------

    inline void ExecutionEngine::execute(const IRModule& module) {
        execute(module.program(), module.unitaries());
    }

} // namespace qlassical::backend

#endif // QLASSICAL_BACKEND_CPU_OPENMP_ENGINE_HPP
