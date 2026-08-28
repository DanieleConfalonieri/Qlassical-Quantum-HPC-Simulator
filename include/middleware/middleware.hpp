#ifndef QLASSICAL_MIDDLEWARE_HPP
#define QLASSICAL_MIDDLEWARE_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <optional>
#include <vector>
#include <span>
#include <Eigen/Dense>
#include <frontend/frontend.hpp>
#include <middleware/hw_topology.hpp>

namespace qlassical::middleware {

    // --------------------------------------------------
    // Backend - Target Execution Backend Identification
    // --------------------------------------------------
    enum class Backend : uint8_t {
        CPU_OPENMP = 0x00,  
        MPI        = 0x01,  
        GPU        = 0x02   
    };

    using BackendType = Backend;

    // --------------------------------------------------
    // Qubit Windowing Cost Model Interface
    // --------------------------------------------------
    class WindowingCostModel {
    public:
        virtual ~WindowingCostModel() = default;
        
        virtual std::optional<std::pair<uint16_t, uint16_t>> evaluate_swap(
            std::span<const uint32_t> log_frequencies,
            std::span<const uint16_t> log_to_phys,
            std::span<const uint16_t> phys_to_log
        ) const = 0;
        
        virtual uint16_t get_k_safe() const = 0;
    };

    class GreedyCacheCostModel : public WindowingCostModel {
    private:
        uint16_t k_safe;
    public:
        explicit GreedyCacheCostModel(uint16_t safe_limit) : k_safe(safe_limit) {}

        uint16_t get_k_safe() const override { return k_safe; }

        std::optional<std::pair<uint16_t, uint16_t>> evaluate_swap(
            std::span<const uint32_t> log_frequencies,
            std::span<const uint16_t> log_to_phys,
            std::span<const uint16_t> phys_to_log
        ) const override {
            uint16_t p = 0xFFFF;
            uint32_t p_freq = 0;
            for (std::size_t l = 0; l < log_frequencies.size(); ++l) {
                if (log_frequencies[l] > 0) {
                    uint16_t phys = log_to_phys[l];
                    if (phys > k_safe) {
                        p = phys;
                        p_freq = log_frequencies[l];
                        break;
                    }
                }
            }

            if (p == 0xFFFF) return std::nullopt;

            uint16_t v = 0xFFFF;
            uint32_t min_freq = UINT32_MAX;
            for (std::size_t l = 0; l < log_frequencies.size(); ++l) {
                uint16_t phys = log_to_phys[l];
                if (phys <= k_safe) {
                    if (log_frequencies[l] < min_freq) {
                        min_freq = log_frequencies[l];
                        v = phys;
                        if (min_freq == 0) break;
                    }
                }
            }

            if (v != 0xFFFF && p_freq > min_freq) {
                return std::make_pair(p, v);
            }
            return std::nullopt;
        }
    };

    // --------------------------------------------------
    // Data-Oriented DAG Representation
    // --------------------------------------------------
    constexpr uint8_t MAX_GATE_ARITY = 4;
    constexpr uint32_t NULL_NODE = 0xFFFFFFFF;
    constexpr std::size_t MAX_MATRIX_DIM = 1ULL << MAX_GATE_ARITY;
    using GateMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, MAX_MATRIX_DIM, MAX_MATRIX_DIM>;

    struct alignas(64) DAGNode {
        GateInstr instr;
        uint32_t inputs[MAX_GATE_ARITY] = {NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE};
        uint32_t outputs[MAX_GATE_ARITY] = {NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE};
        bool is_fused = false; 
    };

    class DAG {
    public:
        UnitaryPool unitary_pool;
        std::vector<DAGNode> nodes;
        std::vector<GateInstr> rewritten_stream;

        // Builds the DAG in O(N) using a frontier approach
        // Executes reserve() to eliminate dynamic reallocations
        static DAG build(IRModule&& module) {
            DAG dag;
            // The DAG takes ownership of the UnitaryPool from the module (zero-copy)
            dag.unitary_pool = std::move(module.unitaries());

            const auto program = module.program();
            dag.nodes.reserve(program.size());

            // Track the last node index that modified each qubit
            std::vector<uint32_t> frontier(module.num_qubits, NULL_NODE);

            for (const auto& instr : program) {
                DAGNode node;
                node.instr = instr;
                node.is_fused = false;
                
                uint32_t current_node_idx = static_cast<uint32_t>(dag.nodes.size());
                
                // Link inputs from the frontier
                uint8_t arity = instr.arity();
                for (uint8_t i = 0; i < arity; ++i) {
                    uint16_t q = instr.qubits[i];
                    uint32_t prev_node_idx = frontier[q];
                    node.inputs[i] = prev_node_idx;
                    
                    // If a previous node exists, update its outputs to point to this node
                    if (prev_node_idx != NULL_NODE) {
                        DAGNode& prev_node = dag.nodes[prev_node_idx];
                        uint8_t prev_arity = prev_node.instr.arity();
                        for (uint8_t j = 0; j < prev_arity; ++j) {
                            if (prev_node.instr.qubits[j] == q) {
                                prev_node.outputs[j] = current_node_idx;
                                break;
                            }
                        }
                    }
                    
                    // Update frontier
                    frontier[q] = current_node_idx;
                }
                
                dag.nodes.push_back(node);
            }
            
            return dag;
        }

        // Flattens the DAG back into a linear instruction stream, bypassing fused nodes
        IRModule release(uint32_t num_qubits) && {
            IRModule optimized_module(num_qubits);
            
            if (!rewritten_stream.empty()) {
                optimized_module.gate_stream = std::move(rewritten_stream);
            } else {
                optimized_module.gate_stream.reserve(nodes.size());
                for (const auto& node : nodes) {
                    if (!node.is_fused) {
                        optimized_module.gate_stream.push_back(node.instr);
                    }
                }
            }
            
            // Move the updated unitary pool into the reconstructed module
            optimized_module.unitary_pool = std::move(unitary_pool);
            return optimized_module;
        }
    };

    // --------------------------------------------------
    // Isolated Gate Fusion Math (Eigen Kronecker Product)
    // --------------------------------------------------
    namespace FusionMath {
        // High-Performance Branchless Scatter Expansion
        // Strictly Zero-Allocation (Stack only)
        inline GateMatrix expand_matrix(
            const GateMatrix& U,
            std::span<const uint16_t> q_sub,
            std::span<const uint16_t> q_full
        ) {
            std::size_t dim_full = 1ULL << q_full.size();
            std::size_t dim_sub = 1ULL << q_sub.size();
            
            // Determine the un-acted qubits (Q_rest)
            std::array<uint16_t, MAX_GATE_ARITY> q_rest;
            uint8_t q_rest_size = 0;
            for (uint16_t qf : q_full) {
                bool found = false;
                for (auto qs : q_sub) {
                    if (qf == qs) { found = true; break; }
                }
                if (!found) q_rest[q_rest_size++] = qf;
            }
            std::size_t dim_rest = 1ULL << q_rest_size;

            // Pre-compute mapped bits for sub-indices
            std::array<std::size_t, 16> sub_to_full{};
            for (std::size_t r = 0; r < dim_sub; ++r) {
                std::size_t mapped = 0;
                for (std::size_t i = 0; i < q_sub.size(); ++i) {
                    if ((r >> i) & 1) {
                        for (std::size_t pos = 0; pos < q_full.size(); ++pos) {
                            if (q_full[pos] == q_sub[i]) {
                                mapped |= (1ULL << pos);
                                break;
                            }
                        }
                    }
                }
                sub_to_full[r] = mapped;
            }

            // Pre-compute mapped bits for rest-indices
            std::array<std::size_t, 16> rest_to_full{};
            for (std::size_t r = 0; r < dim_rest; ++r) {
                std::size_t mapped = 0;
                for (std::size_t i = 0; i < q_rest_size; ++i) {
                    if ((r >> i) & 1) {
                        for (std::size_t pos = 0; pos < q_full.size(); ++pos) {
                            if (q_full[pos] == q_rest[i]) {
                                mapped |= (1ULL << pos);
                                break;
                            }
                        }
                    }
                }
                rest_to_full[r] = mapped;
            }

            // Pre-allocate the target matrix (Zero initialization)
            GateMatrix M_full = GateMatrix::Zero(dim_full, dim_full);

            // Branchless scatter
            for (std::size_t idx_rest = 0; idx_rest < dim_rest; ++idx_rest) {
                std::size_t base_idx = rest_to_full[idx_rest];
                for (std::size_t r_sub = 0; r_sub < dim_sub; ++r_sub) {
                    std::size_t r_full = base_idx | sub_to_full[r_sub];
                    for (std::size_t c_sub = 0; c_sub < dim_sub; ++c_sub) {
                        std::size_t c_full = base_idx | sub_to_full[c_sub];
                        M_full(r_full, c_full) = U(r_sub, c_sub);
                    }
                }
            }
            return M_full;
        }

        inline GateMatrix fuse_matrices(
            const GateMatrix& U1, std::span<const uint16_t> q1,
            const GateMatrix& U2, std::span<const uint16_t> q2,
            std::array<uint16_t, MAX_GATE_ARITY>& out_qubits,
            uint8_t& out_size
        ) {
            out_size = 0;
            for (auto q : q1) out_qubits[out_size++] = q;
            for (auto q : q2) {
                bool found = false;
                for (uint8_t i = 0; i < out_size; ++i) {
                    if (out_qubits[i] == q) { found = true; break; }
                }
                if (!found) {
                    out_qubits[out_size++] = q;
                }
            }
            std::sort(out_qubits.begin(), out_qubits.begin() + out_size);

            GateMatrix M1 = expand_matrix(U1, q1, std::span<const uint16_t>(out_qubits.data(), out_size));
            GateMatrix M2 = expand_matrix(U2, q2, std::span<const uint16_t>(out_qubits.data(), out_size));
            
            // U2 is applied after U1, so we return M2 * M1
            return M2 * M1;
        }
    }

    // -----------------------------------------------------------
    // Middleware - Hardware-Aware Circuit Transpiler and Optimizer
    // -----------------------------------------------------------
    class Middleware {
    public:
        // Converts Standard Gates into Dense Matrices (Helper)
        static GateMatrix get_matrix(const GateInstr& instr, const UnitaryPool& pool) {
            if (instr.type == GateType::UNITARY || instr.type == GateType::FUSED_BLOCK) {
                auto span = pool.get(instr.payload.matrix_idx);
                std::size_t dim = 1ULL << instr.arity();
                return Eigen::Map<const GateMatrix>(
                    span.data(), dim, dim
                );
            }
            if (instr.type == GateType::H) {
                GateMatrix H(2, 2);
                double inv = 1.0 / std::sqrt(2.0);
                H << inv, inv, inv, -inv;
                return H;
            }
            if (instr.type == GateType::X) {
                GateMatrix X(2, 2); X << 0.0, 1.0, 1.0, 0.0; return X;
            }
            if (instr.type == GateType::Y) {
                GateMatrix Y(2, 2); Y << 0.0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0.0; return Y;
            }
            if (instr.type == GateType::Z) {
                GateMatrix Z(2, 2); Z << 1.0, 0.0, 0.0, -1.0; return Z;
            }
            if (instr.type == GateType::S) {
                GateMatrix S(2, 2); S << 1.0, 0.0, 0.0, std::complex<double>(0, 1); return S;
            }
            if (instr.type == GateType::CX) {
                GateMatrix CX(4, 4);
                CX << 1, 0, 0, 0,
                      0, 0, 0, 1,
                      0, 0, 1, 0,
                      0, 1, 0, 0;
                return CX;
            }
            throw std::runtime_error("Middleware::get_matrix unsupported gate type.");
        }

        static bool is_fusable(const GateInstr& instr) {
            return instr.type == GateType::H || instr.type == GateType::X || instr.type == GateType::Y ||
                   instr.type == GateType::Z || instr.type == GateType::S || instr.type == GateType::CX ||
                   instr.type == GateType::UNITARY || instr.type == GateType::FUSED_BLOCK;
        }

        constexpr Middleware() noexcept = default;

        Middleware(Middleware&&) noexcept = default;
        Middleware& operator=(Middleware&&) noexcept = default;
        Middleware(const Middleware&) = default;
        Middleware& operator=(const Middleware&) = default;
        ~Middleware() = default;

        void transpile(IRModule& module, Backend backend = Backend::CPU_OPENMP) {
            switch (backend) {
                case Backend::CPU_OPENMP:
                    transpile_cpu_openmp(module);
                    break;
                case Backend::MPI:
                    transpile_mpi(module);
                    break;
                case Backend::GPU:
                    transpile_gpu(module);
                    break;
                default:
                    transpile_cpu_openmp(module);
                    break;
            }
        }

    private:
        void transpile_cpu_openmp(IRModule& module) {
            // 1. Build the DAG (O(N), contiguous memory, zero dynamic reallocations)
            DAG dag = DAG::build(std::move(module));
            
            // 2. Aggressive Gate Fusion Pass
            pass_gate_fusion(dag);
            
            // 3. Qubit Windowing Pass (JIT Streamer)
            CPUHwlocTopology topology;
            uint16_t k_safe = topology.get_safe_qubit_limit();
            GreedyCacheCostModel cost_model(k_safe);
            pass_qubit_windowing(dag, cost_model, module.num_qubits);
            
            // 4. Release optimized DAG back to module
            module = std::move(dag).release(module.num_qubits);
        }

        void pass_gate_fusion(DAG& dag) {
            for (std::size_t i = 0; i < dag.nodes.size(); ++i) {
                if (dag.nodes[i].is_fused) continue;
                if (!is_fusable(dag.nodes[i].instr)) continue;
                
                std::vector<uint32_t> block = { static_cast<uint32_t>(i) };
                std::array<uint16_t, MAX_GATE_ARITY> block_q;
                uint8_t block_arity = dag.nodes[i].instr.arity();
                for (uint8_t k = 0; k < block_arity; ++k) block_q[k] = dag.nodes[i].instr.qubits[k];
                bool has_dense = (dag.nodes[i].instr.type == GateType::UNITARY || dag.nodes[i].instr.type == GateType::FUSED_BLOCK);
                
                // Aggressively explore the connected component of fusable gates
                bool found_more = true;
                while (found_more) {
                    found_more = false;
                    for (uint32_t b_cand : block) {
                        for (uint8_t out_idx = 0; out_idx < dag.nodes[b_cand].instr.arity(); ++out_idx) {
                            uint32_t next = dag.nodes[b_cand].outputs[out_idx];
                            if (next == NULL_NODE || dag.nodes[next].is_fused || !is_fusable(dag.nodes[next].instr)) continue;
                            
                            bool already_in_block = false;
                            for (auto n : block) { if (n == next) already_in_block = true; }
                            if (already_in_block) continue;
                            
                            // Causality check: all of `next`'s dependencies must be satisfied
                            bool causality_ok = true;
                            for (uint8_t in_idx = 0; in_idx < dag.nodes[next].instr.arity(); ++in_idx) {
                                uint32_t p = dag.nodes[next].inputs[in_idx];
                                if (p != NULL_NODE && p > i) {
                                    bool p_in_block = false;
                                    for (auto n : block) { if (n == p) p_in_block = true; }
                                    if (!p_in_block) { causality_ok = false; break; }
                                }
                            }
                            if (!causality_ok) continue;
                            
                            // Check combined arity
                            std::array<uint16_t, MAX_GATE_ARITY> union_q;
                            uint8_t union_size = 0;
                            for (uint8_t k = 0; k < block_arity; ++k) union_q[union_size++] = block_q[k];
                            
                            bool union_exceeds = false;
                            for (uint8_t k = 0; k < dag.nodes[next].instr.arity(); ++k) {
                                bool found = false;
                                for (uint8_t u = 0; u < union_size; ++u) {
                                    if (union_q[u] == dag.nodes[next].instr.qubits[k]) { found = true; break; }
                                }
                                if (!found) {
                                    if (union_size >= MAX_GATE_ARITY) { union_exceeds = true; break; }
                                    union_q[union_size++] = dag.nodes[next].instr.qubits[k];
                                }
                            }
                            if (union_exceeds) continue;
                            
                            // Accept the node into the fusion block
                            block.push_back(next);
                            block_arity = union_size;
                            for (uint8_t k = 0; k < union_size; ++k) block_q[k] = union_q[k];
                            if (dag.nodes[next].instr.type == GateType::UNITARY || dag.nodes[next].instr.type == GateType::FUSED_BLOCK) has_dense = true;
                            
                            found_more = true;
                            break; // Restart outer exploration loop
                        }
                        if (found_more) break;
                    }
                }
                
                // --- FUSION HEURISTIC ---
                // Do not commit fusion if it's only 2 standard gates. 
                // A block of 2 standard gates is faster natively than as a Gather/Scatter FUSED_BLOCK.
                if (block.size() == 1) continue;
                if (block.size() == 2 && !has_dense) continue;
                
                // Commit Fusion: Multiply matrices topologically
                GateMatrix M_fused = get_matrix(dag.nodes[block[0]].instr, dag.unitary_pool);
                std::array<uint16_t, MAX_GATE_ARITY> current_q;
                uint8_t current_size = dag.nodes[block[0]].instr.arity();
                for (uint8_t k = 0; k < current_size; ++k) current_q[k] = dag.nodes[block[0]].instr.qubits[k];
                
                for (std::size_t k = 1; k < block.size(); ++k) {
                    GateMatrix mat_B = get_matrix(dag.nodes[block[k]].instr, dag.unitary_pool);
                    std::span<const uint16_t> qA(current_q.data(), current_size);
                    std::array<uint16_t, MAX_GATE_ARITY> qB_arr;
                    for (uint8_t u = 0; u < dag.nodes[block[k]].instr.arity(); ++u) qB_arr[u] = dag.nodes[block[k]].instr.qubits[u];
                    std::span<const uint16_t> qB(qB_arr.data(), dag.nodes[block[k]].instr.arity());
                    
                    std::array<uint16_t, MAX_GATE_ARITY> out_q;
                    uint8_t out_size = 0;
                    
                    M_fused = FusionMath::fuse_matrices(M_fused, qA, mat_B, qB, out_q, out_size);
                    
                    current_size = out_size;
                    for (uint8_t u = 0; u < out_size; ++u) current_q[u] = out_q[u];
                }
                
                // Stitch graph edges
                std::array<uint32_t, MAX_GATE_ARITY> block_inputs;
                std::array<uint32_t, MAX_GATE_ARITY> block_outputs;
                
                for (uint8_t u = 0; u < current_size; ++u) {
                    uint16_t q = current_q[u];
                    
                    uint32_t first_in = NULL_NODE;
                    for (std::size_t k = 0; k < block.size(); ++k) {
                        DAGNode& node = dag.nodes[block[k]];
                        bool found = false;
                        for (uint8_t j = 0; j < node.instr.arity(); ++j) {
                            if (node.instr.qubits[j] == q) {
                                first_in = node.inputs[j];
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                    block_inputs[u] = first_in;
                    
                    uint32_t last_out = NULL_NODE;
                    for (std::size_t k = block.size(); k-- > 0;) {
                        DAGNode& node = dag.nodes[block[k]];
                        bool found = false;
                        for (uint8_t j = 0; j < node.instr.arity(); ++j) {
                            if (node.instr.qubits[j] == q) {
                                last_out = node.outputs[j];
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                    block_outputs[u] = last_out;
                }
                
                // Store in pool
                std::vector<std::complex<double>> m_data(M_fused.data(), M_fused.data() + M_fused.size());
                uint32_t new_matrix_idx = dag.unitary_pool.store(m_data, current_size);
                
                // Morph Node A into the Fused Block
                DAGNode& A = dag.nodes[block[0]];
                A.instr.type = GateType::FUSED_BLOCK;
                A.instr.set_arity(current_size);
                A.instr.payload.matrix_idx = new_matrix_idx;
                
                for (uint8_t u = 0; u < MAX_GATE_ARITY; ++u) {
                    if (u < current_size) {
                        A.instr.qubits[u] = current_q[u];
                        A.inputs[u] = block_inputs[u];
                        A.outputs[u] = block_outputs[u];
                        
                        if (block_inputs[u] != NULL_NODE) {
                            DAGNode& P = dag.nodes[block_inputs[u]];
                            for (uint8_t j = 0; j < P.instr.arity(); ++j) {
                                for (std::size_t k = 0; k < block.size(); ++k) {
                                    if (P.outputs[j] == block[k]) P.outputs[j] = block[0];
                                }
                            }
                        }
                        
                        if (block_outputs[u] != NULL_NODE) {
                            DAGNode& S = dag.nodes[block_outputs[u]];
                            for (uint8_t j = 0; j < S.instr.arity(); ++j) {
                                for (std::size_t k = 0; k < block.size(); ++k) {
                                    if (S.inputs[j] == block[k]) S.inputs[j] = block[0];
                                }
                            }
                        }
                    } else {
                        A.instr.qubits[u] = 255;
                        A.inputs[u] = NULL_NODE;
                        A.outputs[u] = NULL_NODE;
                    }
                }
                
                // Mark absorbed nodes as fused
                for (std::size_t k = 1; k < block.size(); ++k) {
                    dag.nodes[block[k]].is_fused = true;
                }
            }
        }
        
    public:
        void pass_qubit_windowing(DAG& dag, const WindowingCostModel& cost_model, uint32_t num_qubits) {
            // JIT Streamer Pass
            std::vector<uint16_t> log_to_phys(num_qubits);
            std::vector<uint16_t> phys_to_log(num_qubits);
            for(uint32_t i = 0; i < num_qubits; ++i) {
                log_to_phys[i] = i;
                phys_to_log[i] = i;
            }

            dag.rewritten_stream.reserve(dag.nodes.size());
            const std::size_t W = 30; // Look-ahead window size
            std::vector<uint32_t> frequencies(num_qubits, 0);

            // Traverse DAG topologically
            for (std::size_t i = 0; i < dag.nodes.size(); ++i) {
                if (dag.nodes[i].is_fused) continue;

                // Look-ahead to compute logical frequencies
                std::fill(frequencies.begin(), frequencies.end(), 0);
                std::size_t count = 0;
                for (std::size_t j = i; j < dag.nodes.size() && count < W; ++j) {
                    if (dag.nodes[j].is_fused) continue;
                    for (uint8_t k = 0; k < dag.nodes[j].instr.arity(); ++k) {
                        uint16_t q = dag.nodes[j].instr.qubits[k];
                        if (q < num_qubits) {
                            frequencies[q]++;
                        }
                    }
                    count++;
                }
                
                bool needs_swap = false;
                uint16_t safe_limit = cost_model.get_k_safe();
                for (uint8_t k = 0; k < dag.nodes[i].instr.arity(); ++k) {
                    uint16_t q = dag.nodes[i].instr.qubits[k];
                    if (q < num_qubits && log_to_phys[q] > safe_limit) {
                        needs_swap = true;
                        break;
                    }
                }

                // Swap Loop
                if (needs_swap) {
                    auto swap_pair = cost_model.evaluate_swap(frequencies, log_to_phys, phys_to_log);
                    while (swap_pair.has_value()) {
                        uint16_t p = swap_pair->first;
                        uint16_t v = swap_pair->second;
                        
                        // Inject GLOBAL_SWAP instruction
                        GateInstr swap_instr;
                        swap_instr.type = GateType::GLOBAL_SWAP;
                        swap_instr.set_arity(2);
                        swap_instr.qubits[0] = p; // Output targets physical qubits for execution
                        swap_instr.qubits[1] = v;
                        
                        dag.rewritten_stream.push_back(swap_instr);
                        
                        // Update mappings
                        uint16_t log_p = phys_to_log[p];
                        uint16_t log_v = phys_to_log[v];
                        
                        log_to_phys[log_p] = v;
                        log_to_phys[log_v] = p;
                        phys_to_log[p] = log_v;
                        phys_to_log[v] = log_p;

                        swap_pair = cost_model.evaluate_swap(frequencies, log_to_phys, phys_to_log);
                    }
                }

                // Rewrite node.instr.qubits using log_to_phys mapping
                GateInstr rewritten = dag.nodes[i].instr;
                for (uint8_t k = 0; k < rewritten.arity(); ++k) {
                    uint16_t log_q = rewritten.qubits[k];
                    if (log_q < num_qubits) {
                        rewritten.qubits[k] = log_to_phys[log_q];
                    }
                }
                
                // Emit rewritten instruction to the new output stream
                dag.rewritten_stream.push_back(rewritten);
            }
        }

        void transpile_mpi([[maybe_unused]] IRModule& module) {}
        void transpile_gpu([[maybe_unused]] IRModule& module) {}
    };

} // namespace qlassical::middleware

namespace qlassical {
    using middleware::Backend;
    using middleware::BackendType;
    using middleware::Middleware;
} 

#endif // QLASSICAL_MIDDLEWARE_HPP
