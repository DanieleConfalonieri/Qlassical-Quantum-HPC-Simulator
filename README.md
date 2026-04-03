# Qlassical-Quantum-HPC-Simulator
This project tries to create a Quantum Computing simulator, merging different state of the art techniques to achieve optimal performance.

Build and Test

- Configure: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
- Build: cmake --build build 
- Run Tests: ctest --test-dir build --output-on-failure
