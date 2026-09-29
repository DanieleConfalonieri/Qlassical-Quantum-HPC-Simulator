#ifndef QLASSICAL_FRONTEND_HPP
#define QLASSICAL_FRONTEND_HPP

// -------------------------------------------------------------
// Frontend — Quantum Circuit Intermediate Representation & API
// -------------------------------------------------------------
//
// Defines the core data structures and builder interfaces for Qlassical:
// GateType, GateInstr (compact 16-byte IR), UnitaryPool, IRModule, and
// QuantumCircuit builder.

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
    // GateFlags - Bit-packed modifier flags 
    // ------------------------------------- 

    namespace gate_flags {
        inline constexpr uint8_t NONE         = 0x00;
        inline constexpr uint8_t ADJOINT      = 0x01;  // Apply conjugate transpose
        inline constexpr uint8_t FUSED_HEAD   = 0x02;  // First gate of a fused block
        inline constexpr uint8_t FUSED_TAIL   = 0x03;  // Last gate of a fused block
        inline constexpr uint8_t COMPILER_GEN = 0x04;  // Injected by Middle-End, mainly for debugging
    }

    // --------------------------------------------------------
    // GateInstr - The central IR instruction (16-byte aligned)
    // --------------------------------------------------------
    //
    // Memory layout (offsets verified by static_assert):
    //
    //   Byte  0     : GateType      type        (1B)
    //   Byte  1     : uint8_t       flags_arity (1B)
    //   Byte  2-5   : uint8_t[4]    qubits      (4B)
    //   Byte  6-7   : uint16_t      _pad        (2B)
    //   Byte  8-11  : union payload             (4B)
    //   Byte 12-15  : uint32_t      uid         (4B)
    //
    // Total: 16 bytes. Four fit in a cache line.
    // The Middle-End iterates this linearly -> prefetching.

    // Special value for "no matrix" 
    inline constexpr uint32_t NO_MATRIX = UINT32_MAX;

    struct alignas(16) GateInstr {
        GateType type = GateType::H;
        uint8_t  flags_arity = 0;
        uint8_t  qubits[4] = {255, 255, 255, 255};
        
        uint16_t _pad = 0;

        // Shared space: 4 bytes
        union Payload {
            float    param;      // Used when GateType is RX, RY, RZ
            uint32_t matrix_idx; // Used when GateType is UNITARY or FUSED_BLOCK
        } payload = {0.0f};

        uint32_t uid = 0;

        // Helpers
        [[nodiscard]] uint8_t arity() const noexcept { return flags_arity & 0x0F; }
        void set_arity(uint8_t a) noexcept { flags_arity = (flags_arity & 0xF0) | (a & 0x0F); }

        [[nodiscard]] uint8_t flags() const noexcept { return (flags_arity >> 4) & 0x0F; }
        void set_flags(uint8_t f) noexcept { flags_arity = (flags_arity & 0x0F) | ((f & 0x0F) << 4); }
    };

    // -----------------------------------------------------------------
    // UnitaryPool - custom/fused unitary matrices
    // -----------------------------------------------------------------
    //
    // Storage is a flat vector of complex<double>. Each matrix is stored
    // row-major, starting at a known offset. The GateInstr's
    // matrix_idx stores the START INDEX into this flat array, not a pointer.
    //
    // For a k-qubit unitary: size = 2^k × 2^k = 4^k complex entries.

    struct UnitaryPool {
        struct MatrixEntry {
            uint32_t offset;       // Start index into `data`
            uint32_t num_elements; // Number of complex<double> entries (= 4^k)
            uint8_t  num_qubits;   // k (the matrix acts on k qubits)
            uint8_t  _pad[3] = {};
        };

        std::vector<std::complex<double>> data;   
        std::vector<MatrixEntry>          metadata;  

        // Store a new matrix, return its matrix_idx
        // 1) nodiscard -> warn if the return value is ignored
        // 2) span -> light view of a general contiguous sequence of objects, more general than std::vector (C++20)
        //            it contains a pointer to the beginning of the sequence and its length.
        [[nodiscard]] uint32_t store(std::span<const std::complex<double>> matrix,
                                    uint8_t num_qubits) {
            // expected_size = 2^(2k) 
            const uint32_t expected = static_cast<uint32_t>(1u) << (2u * num_qubits);
            if (matrix.size() != expected) {
                throw std::invalid_argument(
                    "UnitaryPool::store: matrix size mismatch. Expected " +
                    std::to_string(expected) + ", got " +
                    std::to_string(matrix.size()));
            }

            const uint32_t idx    = static_cast<uint32_t>(metadata.size());
            const uint32_t offset = static_cast<uint32_t>(data.size());

            data.insert(data.end(), matrix.begin(), matrix.end());
            metadata.push_back(MatrixEntry{offset, expected, num_qubits});

            return idx;
        }

        // Retrieve a matrix as a span (zero-copy) 
        [[nodiscard]] std::span<const std::complex<double>>
        get(uint32_t matrix_idx) const {
            assert(matrix_idx < metadata.size() && "matrix_idx out of bounds");
            const auto& entry = metadata[matrix_idx];
            return {data.data() + entry.offset, entry.num_elements};
        }

        // Query 
        [[nodiscard]] std::size_t num_matrices()    const noexcept { return metadata.size(); }
        [[nodiscard]] std::size_t total_elements()  const noexcept { return data.size(); }
        [[nodiscard]] bool        empty()           const noexcept { return metadata.empty(); }
    };

    // ----------------------------------------------
    // IRModule - Intermediate program representation
    // ----------------------------------------------
    //
    // Owns both the hot (GateInstr) and cold (UnitaryPool) data.

    struct IRModule {
        uint32_t                num_qubits = 0;
        std::vector<GateInstr>  gate_stream;    // hot: linear instruction stream
        UnitaryPool             unitary_pool;   // cold: custom matrices

        IRModule() = default;

        explicit IRModule(uint32_t nq, std::size_t gate_hint = 256)
            : num_qubits(nq)
        {
            gate_stream.reserve(gate_hint);
        }

        // Move-only semantics; allow only move, no deep copy
        IRModule(IRModule&&) noexcept = default;
        IRModule& operator=(IRModule&&) noexcept = default;
        IRModule(const IRModule&) = delete;
        IRModule& operator=(const IRModule&) = delete;

        // Span accessors for compiler passes -> safe access
        [[nodiscard]] std::span<const GateInstr> program()  const noexcept { return gate_stream; }
        [[nodiscard]] std::span<GateInstr>       program()        noexcept { return gate_stream; }

        [[nodiscard]] const UnitaryPool& unitaries() const noexcept { return unitary_pool; }
        [[nodiscard]] UnitaryPool&       unitaries()       noexcept { return unitary_pool; }

        // Query 
        [[nodiscard]] std::size_t gate_count() const noexcept { return gate_stream.size(); }
        [[nodiscard]] bool        empty()      const noexcept { return gate_stream.empty(); }
    };

    // ---------------------------
    // QuantumCircuit - IR Builder
    // ---------------------------
    //
    // Usage:
    //   auto module = QuantumCircuit(3, /*hint=*/128)
    //       .h(0)
    //       .cx(0, 1)
    //       .rx(2, M_PI / 4.0)
    //       .measure({0, 1, 2})
    //       .release();
    //
    // After release(), the circuit is in a moved-from state.
    // The [[nodiscard]] on release() prevents accidental discard of the module.

    class QuantumCircuit {
    public:
        
        explicit QuantumCircuit(uint32_t num_qubits, std::size_t gate_count_hint = 256)
            : module_(num_qubits, gate_count_hint)
            , uid_counter_(0)
        {}

        // Move-only (mirrors IRModule)
        QuantumCircuit(QuantumCircuit&&) noexcept = default;
        QuantumCircuit& operator=(QuantumCircuit&&) noexcept = default;
        QuantumCircuit(const QuantumCircuit&) = delete;
        QuantumCircuit& operator=(const QuantumCircuit&) = delete;

        // Transfer ownership to the transpiler 
        // Must be called on an rvalue: std::move(qc).release()
        [[nodiscard]] IRModule release() noexcept {
            return std::move(module_);
        }

        // Query 
        [[nodiscard]] uint32_t    num_qubits() const noexcept { return module_.num_qubits; }
        [[nodiscard]] std::size_t size()       const noexcept { return module_.gate_count(); }

        // Read-only view of current program (for debugging)
        [[nodiscard]] std::span<const GateInstr> program() const noexcept {
            return module_.program();
        }

        // ------------------
        // SINGLE-QUBIT GATES
        // ------------------

        QuantumCircuit& h(int16_t q) {
            push_1q(GateType::H, q);
            return *this;
        }

        QuantumCircuit& x(int16_t q) {
            push_1q(GateType::X, q);
            return *this;
        }

        QuantumCircuit& y(int16_t q) {
            push_1q(GateType::Y, q);
            return *this;
        }

        QuantumCircuit& z(int16_t q) {
            push_1q(GateType::Z, q);
            return *this;
        }

        QuantumCircuit& s(int16_t q) {
            push_1q(GateType::S, q);
            return *this;
        }

        // ------------------------------
        // SINGLE-QUBIT PARAMETRIC GATES
        // ------------------------------

        QuantumCircuit& rx(int16_t q, double theta) {
            push_1q_param(GateType::RX, q, static_cast<float>(theta));
            return *this;
        }

        QuantumCircuit& ry(int16_t q, double theta) {
            push_1q_param(GateType::RY, q, static_cast<float>(theta));
            return *this;
        }

        QuantumCircuit& rz(int16_t q, double theta) {
            push_1q_param(GateType::RZ, q, static_cast<float>(theta));
            return *this;
        }

        // ---------------
        // TWO-QUBIT GATES
        // ---------------

        QuantumCircuit& cx(int16_t ctrl, int16_t tgt) {
            push_2q(GateType::CX, ctrl, tgt);
            return *this;
        }
        // Alias for cx
        QuantumCircuit& cnot(int16_t ctrl, int16_t tgt) { return cx(ctrl, tgt); }

        QuantumCircuit& swap(int16_t q0, int16_t q1) {
            push_2q(GateType::SWAP, q0, q1);
            return *this;
        }

        // -----------------
        // THREE-QUBIT GATES
        // -----------------

        QuantumCircuit& ccx(int16_t c0, int16_t c1, int16_t tgt) {
            push_3q(GateType::CCX, c0, c1, tgt);
            return *this;
        }
        // Alias for ccx
        QuantumCircuit& toffoli(int16_t c0, int16_t c1, int16_t tgt) {
            return ccx(c0, c1, tgt);
        }

        QuantumCircuit& cswap(int16_t ctrl, int16_t q0, int16_t q1) {
            push_3q(GateType::CSWAP, ctrl, q0, q1);
            return *this;
        }
        // Alias for cswap
        QuantumCircuit& fredkin(int16_t ctrl, int16_t q0, int16_t q1) {
            return cswap(ctrl, q0, q1);
        }

        // --------------
        // CUSTOM UNITARY
        // --------------

        /// Apply an arbitrary k-qubit unitary matrix.
        /// `target_qubits` must have exactly k entries matching the matrix dimension.
        /// The matrix is stored in the cold UnitaryPool; only the index is inlined.
        QuantumCircuit& unitary(std::span<const int16_t> target_qubits,
                                std::span<const std::complex<double>> matrix) {
            const auto k = static_cast<uint8_t>(target_qubits.size());
            if (k == 0 || k > 4) {
                throw std::invalid_argument(
                    "unitary: arity must be 1–4, got " + std::to_string(k));
            }

            const uint32_t midx = module_.unitary_pool.store(matrix, k);

            GateInstr instr = {};
            instr.type       = GateType::UNITARY;
            instr.set_arity(k);
            instr.payload.matrix_idx = midx;
            instr.uid        = uid_counter_++;

            for (uint8_t i = 0; i < k; ++i) {
                validate_qubit(target_qubits[i]);
                instr.qubits[i] = static_cast<uint8_t>(target_qubits[i]);
            }

            module_.gate_stream.push_back(instr);
            return *this;
        }

        // ----------------------------
        // CIRCUIT STRUCTURE DIRECTIVES
        // ----------------------------

        // Insert a scheduling barrier on the specified qubits.
        // An empty list means "barrier on all qubits".
        QuantumCircuit& barrier(std::initializer_list<int16_t> qubits = {}) {
            GateInstr instr = {};
            instr.type  = GateType::BARRIER;
            instr.set_arity(static_cast<uint8_t>(
                std::min<std::size_t>(qubits.size(), 4)));
            instr.uid   = uid_counter_++;

            uint8_t i = 0;
            for (int16_t q : qubits) {
                if (i >= 4) break;
                validate_qubit(q);
                instr.qubits[i++] = static_cast<uint8_t>(q);
            }

            module_.gate_stream.push_back(instr);
            return *this;
        }

        // Measure a single qubit in the computational basis.
        // We only allow measurement in such basis because other basis are
        // easy to achieve using other gates.
        QuantumCircuit& measure(int16_t q) {
            push_1q(GateType::MEASURE, q);
            return *this;
        }

        // Measure multiple qubits.
        QuantumCircuit& measure(std::initializer_list<int16_t> qubits) {
            for (int16_t q : qubits) {
                measure(q);
            }
            return *this;
        }

    private:
        IRModule module_;
        uint32_t uid_counter_;

        // Qubit bounds checking 
        void validate_qubit(int16_t q) const {
            if (q < 0 || q >= 256 || static_cast<uint32_t>(q) >= module_.num_qubits) {
                throw std::out_of_range(
                    "Qubit index " + std::to_string(q) +
                    " out of range [0, min(256, " + std::to_string(module_.num_qubits) + "))");
            }
        }

        // Core push
        void push(const GateInstr& instr) {
            GateInstr to_push = instr;
            to_push.uid = uid_counter_++;
            module_.gate_stream.push_back(to_push);
        }

        // Typed push helpers 

        void push_1q(GateType type, int16_t q) {
            validate_qubit(q);
            GateInstr instr;      
            instr.type = type;
            instr.set_arity(1);
            instr.qubits[0] = static_cast<uint8_t>(q);
            push(instr);
        }

        void push_1q_param(GateType type, int16_t q, float theta) {
            validate_qubit(q);
            GateInstr instr;
            instr.type = type;
            instr.set_arity(1);
            instr.qubits[0] = static_cast<uint8_t>(q);
            instr.payload.param = theta;
            push(instr);
        }

        void push_2q(GateType type, int16_t q0, int16_t q1) {
            validate_qubit(q0);
            validate_qubit(q1);
            if (q0 == q1) {
                throw std::invalid_argument(
                    "Two-qubit gate requires distinct qubits, got " +
                    std::to_string(q0) + " == " + std::to_string(q1));
            }
            GateInstr instr;
            instr.type = type;
            instr.set_arity(2);
            instr.qubits[0] = static_cast<uint8_t>(q0);
            instr.qubits[1] = static_cast<uint8_t>(q1);
            push(instr);
        }

        void push_3q(GateType type, int16_t q0, int16_t q1, int16_t q2) {
            validate_qubit(q0);
            validate_qubit(q1);
            validate_qubit(q2);
            if (q0 == q1 || q0 == q2 || q1 == q2) {
                throw std::invalid_argument(
                    "Three-qubit gate requires distinct qubits");
            }
            GateInstr instr;
            instr.type = type;
            instr.set_arity(3);
            instr.qubits[0] = static_cast<uint8_t>(q0);
            instr.qubits[1] = static_cast<uint8_t>(q1);
            instr.qubits[2] = static_cast<uint8_t>(q2);
            push(instr);
        }

        void push_4q(GateType type, int16_t q0, int16_t q1, int16_t q2, int16_t q3) {
            validate_qubit(q0);
            validate_qubit(q1);
            validate_qubit(q2);
            validate_qubit(q3);
            if (q0 == q1 || q0 == q2 || q0 == q3 || q1 == q2 || q1 == q3 || q2 == q3) {
                throw std::invalid_argument(
                    "Four-qubit gate requires distinct qubits");
            }
            GateInstr instr;
            instr.type = type;
            instr.set_arity(4);
            instr.qubits[0] = static_cast<uint8_t>(q0);
            instr.qubits[1] = static_cast<uint8_t>(q1);
            instr.qubits[2] = static_cast<uint8_t>(q2);
            instr.qubits[3] = static_cast<uint8_t>(q3);
            push(instr);
        }
    };

    // -------
    // Helpers  
    // -------

    // gate arity from GateType for standard gates
    [[nodiscard]] constexpr uint8_t gate_arity(GateType type) noexcept {
        const auto v = static_cast<uint8_t>(type);
        // Arity is encoded in the high part of the enum value for standard gates
        if (v < 0x20) return 1;  // Single-qubit (0x0X, 0x1X)
        if (v < 0x30) return 2;  // Two-qubit   (0x2X)
        if (v < 0x40) return 3;  // Three-qubit  (0x3X)
        // Special gates arity is context-dependent
        return 0;
    }

    // Returns true if the gate type requires parameters.
    [[nodiscard]] constexpr bool gate_is_parametric(GateType type) noexcept {
        const auto v = static_cast<uint8_t>(type);
        return (v >= 0x10 && v < 0x20);   // 1Q parametric
    }

    // Returns true if the gate references the UnitaryPool.
    [[nodiscard]] constexpr bool gate_uses_matrix(GateType type) noexcept {
        return type == GateType::UNITARY || type == GateType::FUSED_BLOCK;
    }

    // Returns true if the gate is a compiler directive (not a physical gate).
    [[nodiscard]] constexpr bool gate_is_directive(GateType type) noexcept {
        const auto v = static_cast<uint8_t>(type);
        return v >= 0xE0;
    }


} // namespace qlassical

#endif // QLASSICAL_FRONTEND_HPP