#ifndef QLASSICAL_MIDDLEWARE_HPP
#define QLASSICAL_MIDDLEWARE_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include <span>
#include <frontend/frontend.hpp>

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
        
        virtual bool should_swap(
            std::span<const uint16_t> active_logical_qubits,
            std::span<const uint16_t> logical_to_physical
        ) const = 0;
    };

    // --------------------------------------------------
    // Data-Oriented DAG Representation
    // --------------------------------------------------
    constexpr uint8_t MAX_GATE_ARITY = 4;
    constexpr uint32_t NULL_NODE = 0xFFFFFFFF;

    struct alignas(64) DAGNode {
        GateInstr instr;
        uint32_t inputs[MAX_GATE_ARITY] = {NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE};
        uint32_t outputs[MAX_GATE_ARITY] = {NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE};
        bool is_fused = false; 
    };

    class DAG {
    public:
        // Builds the DAG in O(N) using a frontier approach
        // Executes reserve() to eliminate dynamic reallocations
        static DAG build(const IRModule& module) {
            DAG dag;
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
        IRModule release(uint32_t num_qubits) {
            // Note: In a complete implementation, this would stitch the instructions 
            // back into an IRModule. Currently IRModule is built via QuantumCircuit.
            // This represents the structural skeleton of the final release.
            IRModule optimized_module(num_qubits);
            
            for (const auto& node : nodes) {
                if (!node.is_fused) {
                    // optimized_module.append(node.instr);
                }
            }
            return optimized_module;
        }

        std::vector<DAGNode> nodes;
    };

    // -----------------------------------------------------------
    // Middleware - Hardware-Aware Circuit Transpiler and Optimizer
    // -----------------------------------------------------------
    class Middleware {
    public:
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
            DAG dag = DAG::build(module);
            
            // 2. Aggressive Gate Fusion Pass
            pass_gate_fusion(dag, module);
            
            // 3. Qubit Windowing Pass (JIT Streamer)
            // Example concrete cost model would be passed here
            // pass_qubit_windowing(dag, cost_model, module);
            
            // 4. Release optimized DAG back to module (Skeleton)
            // module = std::move(dag).release(module.num_qubits());
        }

        void pass_gate_fusion(DAG& dag, IRModule& module) {
            // Traverse topologically (which is just a linear scan of the DAG vector)
            for (std::size_t i = 0; i < dag.nodes.size(); ++i) {
                if (dag.nodes[i].is_fused) continue;
                
                // TODO: Aggressive Fusion Logic
                // 1. Examine successor nodes (via dag.nodes[i].outputs).
                // 2. Compute union of logical qubits: union(node_A.qubits, node_B.qubits).
                // 3. If union.size() <= MAX_GATE_ARITY (4):
                //    - Math: Compute Kronecker product of matrices sequentially in-place.
                //    - Store new matrix in module's UnitaryPool.
                //    - Assign new matrix_idx to node_A.
                //    - Update node_A's qubit footprint to the union.
                //    - Update graph edges to bypass node_B.
                //    - Mark node_B as is_fused = true.
            }
        }
        
        void pass_qubit_windowing(DAG& dag, const WindowingCostModel& cost_model, IRModule& module) {
            // JIT Streamer Pass
            std::vector<uint16_t> logical_to_physical(module.num_qubits);
            for(uint16_t i = 0; i < module.num_qubits; ++i) logical_to_physical[i] = i;

            // Traverse DAG topologically
            for (const auto& node : dag.nodes) {
                if (node.is_fused) continue;

                // Evaluate upcoming working set of active logical qubits
                // std::vector<uint16_t> active_qubits = ...
                
                /*
                if (cost_model.should_swap(active_qubits, logical_to_physical)) {
                    // Inject GLOBAL_SWAP operations to move active qubits to physical indices 0-4
                    // Update logical_to_physical mapping table
                }
                */

                // Rewrite node.instr.qubits using logical_to_physical mapping
                // Emit rewritten instruction to the new output stream
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
