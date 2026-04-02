#ifndef QLASSICAL_FRONTEND_HPP
#define QLASSICAL_FRONTEND_HPP

#include <complex>
#include <vector>
#include <span>
#include <cstring>
#include <stdexcept>
#include <cassert>

namespace qlassical {

    // --------------------
    // GateType - (uint8_t)
    // --------------------
    // Layout: ordered by increasing -arity.

    enum class GateType : uint8_t {
        // Single-qubit gates (arity 1)
        H       = 0x00,
        X       = 0x01,
        Y       = 0x02,
        Z       = 0x03,
        S       = 0x04,
        // Single-qubit parametric (arity 1, params > 0) 
        RX      = 0x10,
        RY      = 0x11,
        RZ      = 0x12,
        // Two-qubit gates (arity 2)
        CX      = 0x20,   // CNOT
        SWAP    = 0x21,
        // Three-qubit gates (arity 3) 
        CCX     = 0x30,   // Toffoli
        CSWAP   = 0x31,   // Fredkin
        // Custom / Extensibility 
        UNITARY = 0x80,   // Arbitrary unitary → matrix_idx into UnitaryPool
        // Compiler directives (Middle-End generated) 
        FUSED_BLOCK  = 0xE0,  // Fused gate block → matrix_idx into UnitaryPool
        GLOBAL_SWAP  = 0xE1,  // Physical qubit reordering (streaming permutation)
        // Circuit structure 
        BARRIER = 0xF0,   // Scheduling barrier (no-op in execution)
        MEASURE = 0xF1,   // Measurement (collapses qubit)
    };

    namespace gate_flags {
        inline constexpr uint8_t NONE         = 0x00;
        inline constexpr uint8_t ADJOINT      = 0x01;  // Apply gate† (conjugate transpose)
        inline constexpr uint8_t FUSED_HEAD   = 0x02;  // First gate of a fused block
        inline constexpr uint8_t FUSED_TAIL   = 0x03;  // Last gate of a fused block
        inline constexpr uint8_t COMPILER_GEN = 0x04;  // Injected by Middle-End, mainly for debugging
    }

}

#endif // QLASSICAL_FRONTEND_HPP