#ifndef JACOBI_CUDA_H
#define JACOBI_CUDA_H

#include "grid.h"

/* Default solver parameters */
#define DEFAULT_N        100
#define DEFAULT_MAX_ITER 10000
#define DEFAULT_TOL      1e-6

/* CUDA block dimensions for kernel execution */
#define CUDA_BLOCK_SIZE  16

/* Result struct returned by the solver */
typedef struct {
    int    iterations;   /* number of iterations performed */
    double final_error;  /* residual error at termination  */
    int    converged;    /* 1 if tolerance was met, 0 if max_iter hit */
} CudaJacobiResult;

/*
 * CUDA-accelerated Jacobi iteration.
 *
 *   host_grid  – initialised N×N grid on CPU (boundary already set)
 *   N          – grid dimension
 *   max_iter   – iteration cap
 *   tol        – convergence tolerance (max absolute change per sweep)
 *
 * Returns a CudaJacobiResult with iteration count, final error, and
 * a convergence flag. The result grid is copied back to host_grid.
 */
CudaJacobiResult jacobi_solve_cuda(double **host_grid, int N, int max_iter, double tol);

#endif /* JACOBI_CUDA_H */
