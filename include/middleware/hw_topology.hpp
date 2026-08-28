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
        virtual std::size_t get_l1_cache_size() const = 0;
        virtual uint32_t get_numa_nodes_count() const = 0;

        // Compute the maximum number of qubits that fit entirely in L1 cache
        // k_safe = floor(log2(C / 16)), where 16 represents sizeof(std::complex<double>)
        uint16_t get_safe_qubit_limit() const {
            std::size_t cache_bytes = get_l1_cache_size();
            if (cache_bytes < 16) return 0;
            return static_cast<uint16_t>(std::floor(std::log2(static_cast<double>(cache_bytes) / 16.0)));
        }
    };

    // 2. Concrete Class: CPUHwlocTopology
#if defined(ENABLE_HWLOC)
    class CPUHwlocTopology : public HardwareTopology {
    private:
        hwloc_topology_t topology;
        std::size_t l1_size = 0;
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

            // Query HWLOC_OBJ_L1CACHE
            hwloc_obj_t l1 = hwloc_get_obj_by_type(topology, HWLOC_OBJ_L1CACHE, 0);
            if (l1 && l1->attr) {
                l1_size = l1->attr->cache.size;
            } else {
                // Fallback to typical 32KB if querying fails
                l1_size = 32768; 
            }

            // Query HWLOC_OBJ_NUMANODE
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

        std::size_t get_l1_cache_size() const override { return l1_size; }
        uint32_t get_numa_nodes_count() const override { return numa_nodes; }
    };
#else
    // Fallback if hwloc is not available on the system during build
    class CPUHwlocTopology : public HardwareTopology {
    public:
        std::size_t get_l1_cache_size() const override { return 32768; } // 32KB fallback
        uint32_t get_numa_nodes_count() const override { return 1; }
    };
#endif

    // 3. Concrete Class: GPUCudaTopology (Stub)
    class GPUCudaTopology : public HardwareTopology {
    public:
        std::size_t get_l1_cache_size() const override {
            // TODO: Use cudaGetDeviceProperties to dynamically query the L1/Shared Memory size
            return 16384; // e.g., 16KB fallback for shared memory
        }
        
        uint32_t get_numa_nodes_count() const override {
            // GPUs don't map to NUMA nodes identically, but we can treat it as 1 local domain for now
            return 1; 
        }
    };

} // namespace qlassical::middleware

#endif // QLASSICAL_HW_TOPOLOGY_HPP
