#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <middleware/middleware.hpp>
#include <Eigen/Dense>
#include <vector>

using namespace qlassical::middleware;
using namespace qlassical::middleware::FusionMath;

TEST_CASE("Isolated Gate Fusion Math (Eigen)", "[middleware][fusion]") {
    SECTION("Identical Footprints") {
        // U1 on q0, U2 on q0
        GateMatrix U1(2, 2);
        U1 << 0.0, 1.0,
              1.0, 0.0; // X gate
              
        GateMatrix U2(2, 2);
        U2 << 1.0, 0.0,
              0.0, -1.0; // Z gate
              
        std::vector<uint16_t> q1 = {0};
        std::vector<uint16_t> q2 = {0};
        
        std::array<uint16_t, 4> out_q;
        uint8_t out_size = 0;
        GateMatrix fused = fuse_matrices(U1, q1, U2, q2, out_q, out_size);
        
        REQUIRE(out_size == 1);
        REQUIRE(out_q[0] == 0);
        
        // Z * X = [0, 1; 1, 0] * [1, 0; 0, -1] = [0, 1; -1, 0] wait!
        // U2 * U1 = Z * X
        // X = |0 1|   Z = |1  0|   Z*X = | 0  1|
        //     |1 0|       |0 -1|         |-1  0|
        
        REQUIRE(std::real(fused(0, 0)) == Catch::Approx(0.0));
        REQUIRE(std::real(fused(0, 1)) == Catch::Approx(1.0));
        REQUIRE(std::real(fused(1, 0)) == Catch::Approx(-1.0));
        REQUIRE(std::real(fused(1, 1)) == Catch::Approx(0.0));
    }
    
    SECTION("Disjoint Footprints (Standard Kronecker)") {
        // U1 on q0 (X), U2 on q1 (Z)
        GateMatrix U1(2, 2);
        U1 << 0.0, 1.0,
              1.0, 0.0;
              
        GateMatrix U2(2, 2);
        U2 << 1.0, 0.0,
              0.0, -1.0;
              
        std::vector<uint16_t> q1 = {0};
        std::vector<uint16_t> q2 = {1};
        
        std::array<uint16_t, 4> out_q;
        uint8_t out_size = 0;
        GateMatrix fused = fuse_matrices(U1, q1, U2, q2, out_q, out_size);
        
        REQUIRE(out_size == 2);
        REQUIRE(out_q[0] == 0);
        REQUIRE(out_q[1] == 1);
        
        // Because of the order, M1 = I (x) X, M2 = Z (x) I
        // However, we must be careful with endianness.
        // out_q is {0, 1}. q0 is bit 0, q1 is bit 1.
        // So M1 expands X on bit 0. M1 = I (on bit 1) (tensor) X (on bit 0).
        // M2 expands Z on bit 1. M2 = Z (on bit 1) (tensor) I (on bit 0).
        // Fused = M2 * M1 = Z (tensor) X.
        
        // Z (tensor) X = 
        // | 1*X  0*X | = | 0  1  0  0|
        // | 0*X -1*X |   | 1  0  0  0|
        //                | 0  0  0 -1|
        //                | 0  0 -1  0|
        
        REQUIRE(std::real(fused(0, 1)) == Catch::Approx(1.0));
        REQUIRE(std::real(fused(1, 0)) == Catch::Approx(1.0));
        REQUIRE(std::real(fused(2, 3)) == Catch::Approx(-1.0));
        REQUIRE(std::real(fused(3, 2)) == Catch::Approx(-1.0));
        REQUIRE(std::real(fused(0, 0)) == Catch::Approx(0.0));
        REQUIRE(std::real(fused(3, 3)) == Catch::Approx(0.0));
    }
    
    SECTION("Partial Overlap Expansion") {
        // U1 on q0,q1 (CNOT q0->q1)
        // bit 0 = q0, bit 1 = q1
        GateMatrix U1(4, 4);
        U1 << 1, 0, 0, 0,
              0, 0, 0, 1,
              0, 0, 1, 0,
              0, 1, 0, 0;
              
        // U2 on q1,q2 (CZ)
        // bit 0 = q1, bit 1 = q2
        GateMatrix U2(4, 4);
        U2 << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0,-1;
              
        std::vector<uint16_t> q1 = {0, 1};
        std::vector<uint16_t> q2 = {1, 2};
        
        std::array<uint16_t, 4> out_q;
        uint8_t out_size = 0;
        GateMatrix fused = fuse_matrices(U1, q1, U2, q2, out_q, out_size);
        
        REQUIRE(out_size == 3);
        REQUIRE(out_q[0] == 0);
        REQUIRE(out_q[1] == 1);
        REQUIRE(out_q[2] == 2);
        
        // The resulting matrix is 8x8.
        REQUIRE(fused.rows() == 8);
        REQUIRE(fused.cols() == 8);
        
        // Let's test a specific column.
        // Apply to |110> = |q2=0, q1=1, q0=1>. 
        // Index is 011_2 = 3.
        // First U1 (CNOT q0->q1): q0=1, so q1 flips. 
        // State becomes |q2=0, q1=0, q0=1> = 001_2 = 1.
        // Then U2 (CZ q1, q2): q1=0, q2=0. No phase flip.
        // Final state = |001> (Index 1).
        
        // So fused * |3> = |1>, meaning fused(1, 3) = 1.0, and others in col 3 are 0.
        REQUIRE(std::real(fused(1, 3)) == Catch::Approx(1.0));
        REQUIRE(std::real(fused(3, 3)) == Catch::Approx(0.0));
        
        // Let's test |111> = |q2=1, q1=1, q0=1>. Index = 7.
        // U1 flips q1: state becomes |101> = 5.
        // U2 CZ on q1=0, q2=1: No phase flip.
        // Final state = |5>.
        // So fused(5, 7) = 1.0
        REQUIRE(std::real(fused(5, 7)) == Catch::Approx(1.0));
        
        // Let's test |101> = |q2=1, q1=0, q0=1>. Index = 5.
        // U1 flips q1: state becomes |111> = 7.
        // U2 CZ on q1=1, q2=1: Phase flip! -1.
        // Final state = -|111> = -|7>.
        // So fused(7, 5) = -1.0
        REQUIRE(std::real(fused(7, 5)) == Catch::Approx(-1.0));
    }
}
