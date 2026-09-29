#ifndef QLASSICAL_BACKEND_STATE_VECTOR_UTILS_HPP
#define QLASSICAL_BACKEND_STATE_VECTOR_UTILS_HPP

// -------------------------------------------------------------
// StateVectorUtils - State Vector Bit Manipulation Utilities
// -------------------------------------------------------------
//
// Utility functions for state vector manipulation across execution backends (CPU, GPU, etc.).
// Follows the DRY principle: since every gate kernel accesses the state vector exploiting
// the bit-level structure of computational basis indices, these inline bitwise operations
// are shared across backends and gates.

#include <cstddef>
#include <cstdint>

namespace qlassical::backend::utils {

    // ---------------------------------------------------------
    // Bit Manipulation Functions
    // ---------------------------------------------------------

    // Inserts a zero bit at position `target`, shifting bits at and above `target` to the left.
    [[nodiscard]] constexpr inline std::size_t insert_zero_bit(std::size_t idx, uint8_t target) noexcept {
        // Mask for the bits to the right of the target position
        const std::size_t mask = (std::size_t{1} << target) - 1;
        // Insert a zero bit at the target position by splitting the index into two parts
        return (idx & mask) | ((idx & ~mask) << 1);
    }

    // Flips the bit at position `target`.
    [[nodiscard]] constexpr inline std::size_t flip_target_bit(std::size_t idx, uint8_t target) noexcept {
        // Flip the bit at the target position
        return idx ^ (std::size_t{1} << target);
    }

} // namespace qlassical::backend::utils

#endif // QLASSICAL_BACKEND_STATE_VECTOR_UTILS_HPP
