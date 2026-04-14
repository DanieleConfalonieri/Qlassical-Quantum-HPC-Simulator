#ifndef QLASSICAL_BACKEND_STATE_VECTOR_UTILS_HPP
#define QLASSICAL_BACKEND_STATE_VECTOR_UTILS_HPP
    // --------------------------------------------------
    // StateVectorUtils - State Vector Utility Functions
    // --------------------------------------------------

    // Utility functions for state vector manipulation, eventually used by multiple backends (CPU, GPU, etc.)
    // The main purpose of this header is to follow the DRY principle. Since every gate will access the state vector
    // exploiting the bit-level trick that exploits the regularity of the binary encoding, the inline implementation
    // of such bitwise operations (better explained in the doc) will be shared across backends and gates.

    #include <cstdint>

    namespace qlassical::backend::utils {

        [[nodiscard]] constexpr inline std::size_t insert_zero_bit(std::size_t idx, int16_t target) noexcept {
            // Mask for the bits to the right of the target position
            const std::size_t mask = (std::size_t{1} << target) - 1;
            // Insert a zero bit at the target position by splitting the index into two parts
            return (idx & mask) | ((idx & ~mask) << 1);
        }

        [[nodiscard]] constexpr inline std::size_t flip_target_bit(std::size_t idx, int16_t target) noexcept {
            // Flip the bit at the target position
            return idx ^ (std::size_t{1} << target);
        }

    }

#endif 