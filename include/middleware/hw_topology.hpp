#ifndef QLASSICAL_HW_TOPOLOGY_HPP
#define QLASSICAL_HW_TOPOLOGY_HPP

#include <cstdint>
#include <cmath>
#include <memory>
#include <stdexcept>

// hwloc is only required for CPU topologies
#if defined(ENABLE_HWLOC)
#include <hwloc.h>
#endif

namespace qlassical::middleware {

    // 1. Abstract Base Class
    class HardwareTopology {
    public:
        virtual ~HardwareTopology() = default;

        // Retrieve critical metrics
        virtual std::size_t get_numa_memory_size_bytes() const = 0;
        virtual uint32_t get_numa_nodes_count() const = 0;

        // Compute the maximum number of qubits that fit entirely in a NUMA domain
        // k_safe = floor(log2(M / 16)), where 16 represents sizeof(std::complex<double>)
        uint16_t get_safe_qubit_limit() const {
            /*
            std::size_t memory_bytes = get_numa_memory_size_bytes();
            if (memory_bytes < 16) return 0;
            // 15% of the memory is reserved for system overhead, leaving 85% usable for qubit storage
            double usable_memory = static_cast<double>(memory_bytes) * 0.85;
            return static_cast<uint16_t>(std::floor(std::log2(usable_memory / 16.0)));
            */
            // Problem: swaps were not being injected because k_safe was too high...the RAM wall hits before the NUMA one. 
            // Let us hardcode k_safe = 18 to see the effect of the SWAP injection on the final transpiled circuit size.
            return 18;
        }
    };

    // 2. Concrete Class: CPUHwlocTopology
#if defined(ENABLE_HWLOC)
    class CPUHwlocTopology : public HardwareTopology {
    private:
        hwloc_topology_t topology;
        std::size_t numa_memory_size = 0;
        uint32_t numa_nodes = 0;

    public:
        CPUHwlocTopology() {
            // Safely initialize the hwloc topology tree
            if (hwloc_topology_init(&topology) < 0) {
                throw std::runtime_error("Failed to initialize hwloc topology");
            }
            if (hwloc_topology_load(topology) < 0) {
                hwloc_topology_destroy(topology);
                throw std::runtime_error("Failed to load hwloc topology");
            }

            // Query HWLOC_OBJ_NUMANODE for local memory size
            hwloc_obj_t numa = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, 0);
            if (numa && numa->attr) {
                numa_memory_size = numa->attr->numanode.local_memory;
            } else {
                // Fallback to typical 4 GB if querying fails
                numa_memory_size = 4ULL * 1024 * 1024 * 1024; 
            }

            // Query HWLOC_OBJ_NUMANODE for domain count
            int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
            if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
                numa_nodes = hwloc_get_nbobjs_by_depth(topology, depth);
            }
            if (numa_nodes == 0) numa_nodes = 1; // Fallback to at least 1 NUMA node
        }

        ~CPUHwlocTopology() override {
            // Ensure RAII principles
            hwloc_topology_destroy(topology);
        }

        std::size_t get_numa_memory_size_bytes() const override { return numa_memory_size; }
        uint32_t get_numa_nodes_count() const override { return numa_nodes; }
    };
#else
    // Fallback if hwloc is not available on the system during build
    class CPUHwlocTopology : public HardwareTopology {
    public:
        std::size_t get_numa_memory_size_bytes() const override { return 4ULL * 1024 * 1024 * 1024; } // 4GB fallback
        uint32_t get_numa_nodes_count() const override { return 1; }
    };
#endif

    // 3. Concrete Class: GPUCudaTopology (Stub)
    //class GPUCudaTopology : public HardwareTopology {
    //public:
    //    std::size_t get_numa_memory_size_bytes() const override {
    //        // TODO: Use cudaGetDeviceProperties to dynamically query the global memory size
    //        return 4ULL * 1024 * 1024 * 1024; // e.g., 4GB fallback for global memory
    //    }
    //    
    //    uint32_t get_numa_nodes_count() const override {
    //        // GPUs don't map to NUMA nodes identically, but we can treat it as 1 local domain for now
    //        return 1; 
    //    }
    //};

} // namespace qlassical::middleware

#endif // QLASSICAL_HW_TOPOLOGY_HPP
