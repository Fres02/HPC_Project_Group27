# Project Feature Decomposition

## 1. Serial Core Solver
Build the baseline Jacobi solver.

**Scope**
- Initialize `N x N` grid
- Apply fixed boundary conditions
- Perform Jacobi updates using double buffering
- Compute convergence error
- Stop when tolerance or maximum iterations are reached

**Deliverable**
- Correct serial implementation used as the reference for all other versions.

---

## 2. Grid and Boundary Management
Separate grid setup from solver logic.

**Scope**
- Grid allocation and deallocation
- Boundary condition initialization
- Initial value generation
- Buffer swapping utilities

**Deliverable**
- Reusable module shared across all implementations.

---

## 3. Convergence and Validation
Separate correctness verification.

**Scope**
- Error computation per iteration
- Convergence testing
- Comparison with serial baseline
- Numerical tolerance validation

**Deliverable**
- Validation layer usable by OpenMP, Pthreads, MPI, CUDA, and Hybrid versions.

---

## 4. OpenMP Parallel Version
Shared-memory parallel implementation.

**Scope**
- Parallelize grid update loops
- Handle reductions for error calculation
- Maintain thread-safe iteration flow
- Benchmark against serial version

**Deliverable**
- Standalone OpenMP implementation.

---

## 5. POSIX Threads Version
Low-level threading implementation.

**Scope**
- Divide rows among threads
- Implement barrier synchronization
- Thread-level workload assignment
- Shared convergence checking

**Deliverable**
- Standalone Pthreads implementation.

---

## 6. MPI Distributed Version
Distributed-memory parallel implementation.

**Scope**
- Domain decomposition (row-wise or block-wise)
- Halo exchange between processes
- Local grid computation
- Global convergence checking
- Optional result gathering

**Deliverable**
- Standalone MPI implementation.

---

## 7. CUDA GPU Version
GPU-accelerated implementation.

**Scope**
- Device memory allocation
- Kernel implementation for stencil updates
- Host-device memory transfers
- Pointer swapping on device
- Optional shared-memory optimization

**Deliverable**
- Standalone CUDA implementation.

---

## 8. Hybrid Parallel Version
Combination of multiple parallel techniques.

**Scope**
- MPI across nodes
- OpenMP within each MPI process  
  *or*
- MPI combined with CUDA acceleration
- Integrated convergence and communication strategy

**Deliverable**
- Hybrid parallel implementation.

---

## 9. Performance Benchmarking
Independent benchmarking and analysis.

**Scope**
- Execution time measurement
- Speedup and efficiency calculations
- Scaling tests with different grid sizes
- Comparison across all implementations

**Deliverable**
- Performance evaluation results and graphs.

---

## 10. Experiment Configuration
Separate runtime configuration.

**Scope**
- Grid size configuration
- Maximum iterations
- Convergence tolerance
- Thread count
- MPI process count
- CUDA block configuration

**Deliverable**
- Command-line or configuration file based setup.

---

## 11. Reporting and Comparison
Final project documentation.

**Scope**
- Methodology description
- System architecture explanation
- Correctness validation
- Performance comparison
- Limitations and observations

**Deliverable**
- Final report and presentation materials.

---

# Recommended Implementation Order

1. Serial Core Solver  
2. Grid and Boundary Management  
3. Convergence and Validation  
4. OpenMP Parallel Version  
5. POSIX Threads Version  
6. MPI Distributed Version  
7. CUDA GPU Version  
8. Hybrid Parallel Version  
9. Performance Benchmarking  
10. Reporting  

---

# Suggested Project Structure

```text
project/
├── core/
│   ├── grid
│   ├── boundary
│   ├── jacobi_serial
│   └── convergence
├── openmp/
├── pthreads/
├── mpi/
├── cuda/
├── hybrid/
├── benchmarks/
├── configs/
└── report/
