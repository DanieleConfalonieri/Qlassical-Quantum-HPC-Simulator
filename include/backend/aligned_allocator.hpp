#ifndef QLASSICAL_BACKEND_ALIGNED_ALLOCATOR_HPP
#define QLASSICAL_BACKEND_ALIGNED_ALLOCATOR_HPP

// -----------------------------------------------------------------
// AlignedAllocator — Strict Memory Alignment Allocator
// -----------------------------------------------------------------
//
// A stateless C++ helper allocator that enforces strict memory alignment.
// Default alignment is 64 bytes (cache line boundary, AVX-512 register width).
// This is critical for Eigen::Map<VectorXcd> which requires alignment for
// high-performance vectorized SIMD operations.
//
// Usage:
//   std::vector<std::complex<double>, AlignedAllocator<std::complex<double>>> sv;
//
// The allocator is stateless (no member data), so:
//   - All instances of AlignedAllocator<T, N> are interchangeable (operator== is true).
//   - std::vector move/swap across allocator instances works without conflict.

#include <cstddef>
#include <cstdlib>
#include <new>
#include <type_traits>

namespace qlassical::backend {

    template<typename T, std::size_t Alignment = 64>
    class AlignedAllocator {
        // Alignment must be a power of two
        static_assert((Alignment & (Alignment - 1)) == 0, 
                      "Alignment must be a power of two");

    public:
        // ----------------------
        // Required Type Aliases
        // ----------------------
        using value_type = T;
        using size_type  = std::size_t;

        // Rebinding support: allows std::vector internals to allocate
        // different types using the same allocator.
        template<typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        // ------------
        // Constructors
        // ------------
        constexpr AlignedAllocator() noexcept = default;

        template<typename U>
        constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        // ---------------
        // Core Allocation
        // ---------------
        [[nodiscard]] T* allocate(std::size_t n) {
            if (n == 0) return nullptr;

            // Overflow check: n * sizeof(T) must not overflow size_t
            if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
                throw std::bad_alloc();
            }

            const std::size_t byte_count = n * sizeof(T);
            void* ptr = nullptr;

#if defined(_MSC_VER)
            ptr = _aligned_malloc(byte_count, Alignment);
            if (!ptr) throw std::bad_alloc();
#else
            // posix_memalign requires size >= sizeof(void*) and alignment
            // to be a power of two AND a multiple of sizeof(void*).
            const int ret = posix_memalign(&ptr, Alignment, byte_count);
            if (ret != 0) throw std::bad_alloc();
#endif

            return static_cast<T*>(ptr);
        }

        // -----------------
        // Core Deallocation
        // -----------------
        void deallocate(T* ptr, [[maybe_unused]] std::size_t n) noexcept {
            if (!ptr) return;

#if defined(_MSC_VER)
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }

        // ------------------
        // Equality Operators
        // ------------------
        template<typename U>
        [[nodiscard]] constexpr bool operator==(
            const AlignedAllocator<U, Alignment>&) const noexcept {
            return true;
        }

        template<typename U>
        [[nodiscard]] constexpr bool operator!=(
            const AlignedAllocator<U, Alignment>&) const noexcept {
            return false;
        }
    };

} // namespace qlassical::backend

#endif // QLASSICAL_BACKEND_ALIGNED_ALLOCATOR_HPP
