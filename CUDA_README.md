# CUDA Implementation for Jacobi Solver

## Overview

This directory contains a GPU-accelerated implementation of the Jacobi solver using NVIDIA CUDA. The implementation demonstrates efficient GPU utilization for solving Laplace's equation using the iterative Jacobi method.

## Files

- **jacobi_cuda.h** – Header file defining the CUDA Jacobi solver interface
- **jacobi_cuda.cu** – CUDA implementation with kernels and host-device communication
- **test_jacobi_cuda.c** – Comprehensive test suite for the CUDA solver

## Key Features

### 1. **Efficient CUDA Kernels**
- **jacobi_update_kernel**: Performs 5-point stencil updates (Jacobi iteration)
  - Each CUDA thread updates one grid point
  - Reads from `curr` buffer, writes to `next` buffer
  - Computes absolute differences for convergence checking
  
- **reduce_max_error_kernel**: Parallel reduction for finding maximum error
  - Uses shared memory for fast block-level reduction
  - Implements efficient butterfly reduction pattern

### 2. **Double Buffering on GPU**
- Maintains two device buffers for stencil updates
- Pointer swapping on device (O(1) operation)
- Avoids redundant data transfers between iterations

### 3. **GPU Memory Management**
- Automatic allocation/deallocation of device memory
- Efficient host-device memory transfers
- Error checking with `CUDA_CHECK` macro

### 4. **Convergence Checking**
- Error workspace to store per-point differences
- Parallel reduction to find maximum error
- Early termination when tolerance is reached

## Configuration

### CUDA Block and Grid Dimensions
```c
#define CUDA_BLOCK_SIZE  16      // 16x16 = 256 threads per block
```

The grid layout uses 2D blocks of 16×16 threads for optimal memory coalescing.

### Compute Capability
Default architecture target: `sm_60` (Maxwell and newer)

To use a different architecture, modify in `Makefile`:
```makefile
NVCCFLAGS = -O2 -arch=sm_70    # For Volta
# or
NVCCFLAGS = -O2 -arch=sm_80    # For Ampere
```

## Building

### Prerequisites
- NVIDIA CUDA Toolkit (version 11.0 or later)
- NVIDIA GPU with CUDA compute capability 3.0 or higher
- Standard C compiler (gcc/clang)

### Compilation

From the project root:

```bash
# Build all CUDA components
cd core && make all

# Build and run tests
cd tests && make test
```

Or compile the test directly:
```bash
nvcc -O2 -arch=sm_60 -c jacobi_cuda.cu
gcc -O2 -c grid.c
gcc -O2 test_jacobi_cuda.c grid.o jacobi_cuda.o -lm -lcudart -o test_jacobi_cuda
./test_jacobi_cuda
```

## Usage Example

```c
#include "jacobi_cuda.h"
#include "grid.h"

// Initialize grid on CPU
double **grid = alloc_grid(N);
init_grid(grid, N);

// Run CUDA solver
CudaJacobiResult result = jacobi_solve_cuda(grid, N, 10000, 1e-6);

// Access results
printf("Iterations: %d\n", result.iterations);
printf("Final Error: %g\n", result.final_error);
printf("Converged: %s\n", result.converged ? "YES" : "NO");

// grid now contains the solution (copied back from GPU)
free_grid(grid, N);
```

## API Reference

### Function: `jacobi_solve_cuda`

```c
CudaJacobiResult jacobi_solve_cuda(double **host_grid, int N, 
                                   int max_iter, double tol)
```

**Parameters:**
- `host_grid` – Initialized N×N grid (boundary conditions set)
- `N` – Grid dimension
- `max_iter` – Maximum number of iterations
- `tol` – Convergence tolerance

**Returns:**
- `CudaJacobiResult` struct with:
  - `iterations` – Number of iterations performed
  - `final_error` – Maximum absolute change at convergence
  - `converged` – 1 if tolerance reached, 0 otherwise

**Notes:**
- The input grid is modified in-place with the solution
- All GPU memory is automatically freed upon completion
- Detailed iteration progress is printed to stderr

## Performance Considerations

### 1. **Memory Coalescing**
- Grid data stored in row-major (C-style) order
- 2D thread blocks align with global memory access patterns
- Minimizes memory latency

### 2. **Shared Memory Usage**
- Reduction kernel uses shared memory for efficient max computation
- Block-level reduction reduces global memory traffic

### 3. **Register Usage**
- Stencil kernel uses ~30 registers per thread
- Allows high occupancy on modern GPUs

### 4. **Optimization Tips**
- Use larger block sizes (32x32) for GPUs with higher occupancy
- Consider using unified memory for automatic host-device sync
- Profile with `nvprof` or `nsys` to identify bottlenecks

## Test Suite

The test file `test_jacobi_cuda.c` includes:

1. **test_cuda_convergence** – Verifies solution converges to tolerance
2. **test_cuda_boundary_preservation** – Ensures boundary conditions remain fixed
3. **test_cuda_solution_range** – Validates interior values stay in valid range
4. **test_cuda_various_grid_sizes** – Tests N = 32, 64, 128
5. **test_cuda_error_monotonicity** – Confirms error generally decreases

## Troubleshooting

### CUDA Error: "no CUDA-capable device"
```bash
# Check available GPUs
nvidia-smi

# Or verify CUDA installation
nvcc --version
```

### Compilation Error: "cudaGetErrorString not found"
Ensure CUDA runtime library is linked:
```bash
gcc ... -lcudart
```

### Memory Error: "out of memory"
Reduce grid size or available GPU memory:
```bash
nvidia-smi  # Check available memory
```

## Differences from Serial Implementation

| Aspect | Serial | CUDA |
|--------|--------|------|
| **Location** | `jacobi_serial.c` | `jacobi_cuda.cu` |
| **Stencil Update** | CPU loop | CUDA kernel |
| **Error Calculation** | Sequential max | Parallel reduction |
| **Memory** | Host (CPU) | Device (GPU) |
| **Convergence Check** | Every iteration | Reduced from GPU |
| **Data Transfer** | None | Start & end only |

## Future Enhancements

- [ ] Implement optimized stencil using shared memory (halos)
- [ ] Multi-GPU support with NCCL
- [ ] Mixed-precision computation (FP32 interior, FP64 error tracking)
- [ ] Persistent kernel for better scaling
- [ ] Performance comparison against OpenMP/MPI versions

## References

- NVIDIA CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Jacobi Method: https://en.wikipedia.org/wiki/Jacobi_method
- GPU Optimization: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

