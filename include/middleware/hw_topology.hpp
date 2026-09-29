#ifndef QLASSICAL_HW_TOPOLOGY_HPP
#define QLASSICAL_HW_TOPOLOGY_HPP

// -------------------------------------------------------------
// HardwareTopology — Hardware Architecture & NUMA Introspection
// -------------------------------------------------------------
//
// Discovers hardware topology, memory hierarchies, and NUMA node domains
// using hwloc to guide middleware circuit windowing and SWAP placement.

#include <cstdint>
#include <cmath>
#include <memory>
#include <stdexcept>

// hwloc is only required for CPU topologies
#if defined(ENABLE_HWLOC)
#include <hwloc.h>
#endif

namespace qlassical::middleware {

    // ---------------------------------------------------------
    // HardwareTopology — Abstract Hardware Topology Interface
    // ---------------------------------------------------------

    class HardwareTopology {
    public:
        virtual ~HardwareTopology() = default;

        // -----------------
        // Metric Accessors
        // -----------------

        // Total memory size of a single NUMA node in bytes.
        virtual std::size_t get_numa_memory_size_bytes() const = 0;

        // Total count of NUMA nodes present on the host system.
        virtual uint32_t get_numa_nodes_count() const = 0;

        // -----------------
        // Safe Qubit Limit
        // -----------------

        // Computes the maximum number of qubits that fit entirely within a NUMA domain.
        // Theoretical formula: k_safe = floor(log2(usable_memory / sizeof(complex<double>))).
        // For benchmarking purposes, k_safe is set to 18 to trigger NUMA-aware SWAP injection
        // before encountering the host RAM capacity ceiling.
        uint16_t get_safe_qubit_limit() const {
            return 18;
        }
    };

    // ---------------------------------------------------------
    // CPUHwlocTopology — CPU NUMA Topology Introspection (hwloc)
    // ---------------------------------------------------------

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
                // Fallback to default 4 GB if querying fails
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
    // ---------------------------------------------------------
    // CPUHwlocTopology — Fallback Stub (hwloc Disabled)
    // ---------------------------------------------------------

    class CPUHwlocTopology : public HardwareTopology {
    public:
        std::size_t get_numa_memory_size_bytes() const override { return 4ULL * 1024 * 1024 * 1024; } // 4 GB fallback
        uint32_t get_numa_nodes_count() const override { return 1; }
    };
#endif

    // ---------------------------------------------------------
    // GPUCudaTopology — GPU Topology Stub (Future Expansion)
    // ---------------------------------------------------------
    // class GPUCudaTopology : public HardwareTopology {
    // public:
    //     std::size_t get_numa_memory_size_bytes() const override {
    //         return 4ULL * 1024 * 1024 * 1024; // 4 GB fallback
    //     }
    //     uint32_t get_numa_nodes_count() const override {
    //         return 1;
    //     }
    // };

} // namespace qlassical::middleware

#endif // QLASSICAL_HW_TOPOLOGY_HPP
