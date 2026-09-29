#ifndef QLASSICAL_BACKEND_STATE_VECTOR_HPP
#define QLASSICAL_BACKEND_STATE_VECTOR_HPP

// -------------------------------------------------------------
// CPUStateVector - Contiguous 64-Byte Aligned State Vector (CPU)
// -------------------------------------------------------------
//
// Encapsulates the physical memory for exact state vector simulation.
// The state |psi> is stored as a contiguous array of 2^n complex amplitudes

#include <cmath>
#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "state_vector.hpp"

namespace qlassical::backend {

    class CPUStateVector final : public StateVector {
    public:
        // ----------------------
        // Required Type Aliases
        // ----------------------
        using Amplitude  = std::complex<double>;

        // ------------
        // Constructors
        // ------------
        CPUStateVector() = default;

        // Move-only semantics (mirrors StateVector)
        CPUStateVector(CPUStateVector&&) noexcept = default;
        CPUStateVector& operator=(CPUStateVector&&) noexcept = default;
        CPUStateVector(const CPUStateVector&) = delete;
        CPUStateVector& operator=(const CPUStateVector&) = delete;

        // --------------------------
        // Initialization & Lifecycle
        // --------------------------

        // Allocates 2^num_qubits amplitudes and initializes to |0...0>.
        // Uses OpenMP first-touch policy for NUMA-aware page placement.
        //
        // After this call:
        //   amplitudes_[0] == 1.0 + 0.0i  (the |0...0> amplitude)
        //   amplitudes_[k] == 0.0 + 0.0i  for all k > 0
        //
        // Throws std::bad_alloc if the system cannot satisfy the allocation.
        void initialize(uint32_t num_qubits) override {
            num_qubits_ = num_qubits;
            const std::size_t dim = static_cast<std::size_t>(1) << num_qubits;

            // Allocate raw memory
            amplitudes_.resize(dim);

            // NUMA First-Touch Initialization:
            // On NUMA systems, physical memory pages are allocated on the node
            // of the first thread to write to them. By parallelizing this
            // zeroing loop with static scheduling, we ensure that each thread's
            // pages are allocated on its local NUMA node.
            //
            // schedule(static): deterministic partitioning ensures thread k always
            // owns the same index range, matching the partitioning used by gate kernels.
            const auto dim_signed = static_cast<int64_t>(dim);

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < dim_signed; ++i) {
                amplitudes_[static_cast<std::size_t>(i)] = Amplitude{0.0, 0.0};
            }

            // Set the |0...0> amplitude
            amplitudes_[0] = Amplitude{1.0, 0.0};
        }

        // Allocates and initializes the state vector from an existing amplitude span.
        // Validates that dimension is a power of 2 and amplitudes are normalized.
        void initialize(std::span<const Amplitude> initial_state) override {
            const std::size_t dim = initial_state.size();
            
            // The size must be a non-zero power of 2
            if (dim == 0 || (dim & (dim - 1)) != 0) { 
                throw std::invalid_argument(
                    "CPUStateVector: Input state dimension must be a power of 2.");
            }
            
            num_qubits_ = static_cast<uint32_t>(std::log2(dim));
            
            // Check the normalization condition: sum(|c_i|^2) == 1
            double norm_sq = 0.0;
            const auto dim_signed = static_cast<int64_t>(dim);
            
            #pragma omp parallel for reduction(+:norm_sq) schedule(static)
            for (int64_t i = 0; i < dim_signed; ++i) {
                norm_sq += std::norm(initial_state[static_cast<std::size_t>(i)]);
            }
            
            // Floating point comparison with tolerance; avoid exact equality comparison
            if (std::abs(norm_sq - 1.0) > 1e-6) {
                throw std::invalid_argument(
                    "CPUStateVector: Provided state vector is not normalized (norm squared != 1.0).");
            }
            
            // Actual allocation and NUMA first-touch initialization
            amplitudes_.resize(dim);

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < dim_signed; ++i) {
                amplitudes_[static_cast<std::size_t>(i)] = initial_state[static_cast<std::size_t>(i)];
            }
        }

        // ---------------------------------
        // Span Accessors (Zero-Copy Views)
        // ---------------------------------

        [[nodiscard]] std::span<Amplitude> amplitudes() noexcept {
            return amplitudes_;
        }

        [[nodiscard]] std::span<const Amplitude> amplitudes() const noexcept {
            return amplitudes_;
        }

        // -----------------------------------------
        // Raw Pointer Access (for Eigen::Map)
        // -----------------------------------------
        // Usage:
        //   Eigen::Map<Eigen::VectorXcd> psi(sv.data(), sv.dimension());

        [[nodiscard]] Amplitude* data() noexcept {
            return amplitudes_.data();
        }

        [[nodiscard]] const Amplitude* data() const noexcept {
            return amplitudes_.data();
        }

        // -------------
        // Query Methods
        // -------------

        [[nodiscard]] uint32_t num_qubits() const noexcept override {
            return num_qubits_;
        }

        [[nodiscard]] std::size_t dimension() const noexcept override {
            return amplitudes_.size();
        }

        [[nodiscard]] bool empty() const noexcept override {
            return amplitudes_.empty();
        }

    private:
        std::vector<Amplitude> amplitudes_;
        uint32_t   num_qubits_ = 0;
    };

} // namespace qlassical::backend

#endif // QLASSICAL_BACKEND_STATE_VECTOR_HPP
