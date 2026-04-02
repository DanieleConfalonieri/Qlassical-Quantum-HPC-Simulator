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

    // -------------------------------------
    // GateFlags — Bit-packed modifier flags 
    // ------------------------------------- 

    namespace gate_flags {
        inline constexpr uint8_t NONE         = 0x00;
        inline constexpr uint8_t ADJOINT      = 0x01;  // Apply gate† (conjugate transpose)
        inline constexpr uint8_t FUSED_HEAD   = 0x02;  // First gate of a fused block
        inline constexpr uint8_t FUSED_TAIL   = 0x03;  // Last gate of a fused block
        inline constexpr uint8_t COMPILER_GEN = 0x04;  // Injected by Middle-End, mainly for debugging
    }

    // --------------------------------------------------------
    // GateInstr — The central IR instruction (32-byte aligned)
    // --------------------------------------------------------
    //
    // Memory layout (offsets verified by static_assert):
    //
    //   Byte  0     : GateType      type        (1B)
    //   Byte  1     : uint8_t       arity       (1B)
    //   Byte  2     : uint8_t       flags       (1B)
    //   Byte  3     : uint8_t       _pad0       (1B)
    //   Byte  4–9   : int16_t[3]    qubits      (6B)
    //   Byte 10–11  : uint16_t      _pad1       (2B)
    //   Byte 12–23  : float[3]      params      (12B)
    //   Byte 24–27  : uint32_t      matrix_idx  (4B)
    //   Byte 28–31  : uint32_t      uid         (4B)
    //
    // Total: 32 bytes. Two fit in a cache line.
    // The Middle-End iterates this linearly -> prefetching.

    struct alignas(32) GateInstr {
        GateType  type       = GateType::H;
        uint8_t   arity      = 0;
        uint8_t   flags      = gate_flags::NONE;
        uint8_t   _pad0      = 0;

        int16_t   qubits[3]  = {-1, -1, -1};
        uint16_t  _pad1      = 0;

        float     params[3]  = {0.0f, 0.0f, 0.0f};

        uint32_t  matrix_idx = UINT32_MAX;   // UINT32_MAX ≡ "no matrix"
        uint32_t  uid        = 0;
    };

    // Compile-time layout verification 

    static_assert(sizeof(GateInstr)  == 32, "GateInstr must be exactly 32 bytes");
    static_assert(alignof(GateInstr) == 32, "GateInstr must be 32-byte aligned");

    static_assert(offsetof(GateInstr, type)       == 0,  "type at byte 0");
    static_assert(offsetof(GateInstr, arity)      == 1,  "arity at byte 1");
    static_assert(offsetof(GateInstr, flags)      == 2,  "flags at byte 2");
    static_assert(offsetof(GateInstr, qubits)     == 4,  "qubits at byte 4");
    static_assert(offsetof(GateInstr, params)     == 12, "params at byte 12");
    static_assert(offsetof(GateInstr, matrix_idx) == 24, "matrix_idx at byte 24");
    static_assert(offsetof(GateInstr, uid)        == 28, "uid at byte 28");

    static_assert(std::is_trivially_copyable_v<GateInstr>,
                "GateInstr must be trivially copyable for memcpy/SIMD ops");
    static_assert(std::is_trivially_destructible_v<GateInstr>,
                "GateInstr must be trivially destructible (no cleanup needed)");

    

}

#endif // QLASSICAL_FRONTEND_HPP