#ifndef QLASSICAL_BACKEND_STATE_VECTOR_HPP
#define QLASSICAL_BACKEND_STATE_VECTOR_HPP

// --------------
// CPUStateVector
// --------------

// Encapsulates the physical memory for exact CPUStateVector simulation.
// The state |psi> is stored as a contiguous array of 2^n complex amplitudes
// with 64-byte alignment (enforced by AlignedAllocator).

#include <complex>
#include <cstdint>
#include <span>
#include <vector>

#include "aligned_allocator.hpp"

// Conditionally include OpenMP
#ifdef QLASSICAL_HAS_OPENMP
    #include <omp.h>
#endif

namespace qlassical::backend {

    class CPUStateVector {
    public:
        using Amplitude  = std::complex<double>;
        using AlignedVec = std::vector<Amplitude, AlignedAllocator<Amplitude, 64>>;

        // Default constructor
        CPUStateVector() = default;

        // Move-only
        CPUStateVector(CPUStateVector&&) noexcept = default;
        CPUStateVector& operator=(CPUStateVector&&) noexcept = default;
        CPUStateVector(const CPUStateVector&) = delete;
        CPUStateVector& operator=(const CPUStateVector&) = delete;

        // Initialization

        // Allocates 2^num_qubits amplitudes and initializes to |0...   0>.
        // Uses OpenMP first-touch policy for NUMA-aware page placement.

        // After this call:
        //   amplitudes_[0] == 1.0 + 0.0i  (the |0...0> amplitude)
        //   amplitudes_[k] == 0.0 + 0.0i  for all k > 0

        // Throws std::bad_alloc if the system cannot satisfy the allocation.
        void initialize(uint32_t num_qubits) {
            num_qubits_ = num_qubits;
            const std::size_t dim = static_cast<std::size_t>(1) << num_qubits;

            // Allocate raw memory (AlignedAllocator guarantees 64-byte alignment).
            amplitudes_.resize(dim);

            // ! NUMA First-Touch Initialization !
            //
            // On NUMA systems, physical memory pages are allocated on the node
            // of the first thread to write to them. By parallelizing this
            // zeroing loop with static scheduling, we ensure that each thread's
            // pages are allocated on its local NUMA node.

            // schedule(static): deterministic partitioning -> thread k always
            // owns the same index range. This must match the partitioning used
            // by the gate application kernels so the same thread always accesses
            // the same pages.

            const auto dim_signed = static_cast<int64_t>(dim); // signed to avoid OpenMP warnings

#ifdef QLASSICAL_HAS_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < dim_signed; ++i) {
                amplitudes_[static_cast<std::size_t>(i)] = Amplitude{0.0, 0.0};
            }

            // Set the |0...0> amplitude
            amplitudes_[0] = Amplitude{1.0, 0.0};
        }

        // Span Accessors (zero-copy views) 

        [[nodiscard]] std::span<Amplitude> amplitudes() noexcept {
            return amplitudes_;
        }

        [[nodiscard]] std::span<const Amplitude> amplitudes() const noexcept {
            return amplitudes_;
        }

        // Raw Pointer Access (for Eigen::Map)
        // Usage:
        //   Eigen::Map<Eigen::VectorXcd> psi(sv.data(), sv.dimension());
        // The pointer is guaranteed 64-byte aligned by our custom AlignedAllocator.

        [[nodiscard]] Amplitude* data() noexcept {
            return amplitudes_.data();
        }

        [[nodiscard]] const Amplitude* data() const noexcept {
            return amplitudes_.data();
        }

        // Qury

        [[nodiscard]] uint32_t num_qubits() const noexcept {
            return num_qubits_;
        }

        [[nodiscard]] std::size_t dimension() const noexcept {
            return amplitudes_.size();
        }

        [[nodiscard]] bool empty() const noexcept {
            return amplitudes_.empty();
        }

    private:
        AlignedVec amplitudes_;
        uint32_t   num_qubits_ = 0;
    };

} 
#endif 
