#ifndef JACOBI_PTHREADS_H
#define JACOBI_PTHREADS_H

#include "grid.h"

/* Default solver parameters */
#define DEFAULT_N        100
#define DEFAULT_MAX_ITER 10000
#define DEFAULT_TOL      1e-6
#define DEFAULT_NUM_THREADS 4

/* Result struct returned by the solver */
typedef struct {
    int    iterations;   /* number of iterations performed */
    double final_error;  /* residual error at termination  */
    int    converged;    /* 1 if tolerance was met, 0 if max_iter hit */
} PthreadsJacobiResult;

/*
 * POSIX Threads-based Jacobi iteration.
 *
 *   grid      – initialised N×N grid (boundary already set)
 *   N         – grid dimension
 *   num_threads – number of threads to use
 *   max_iter  – iteration cap
 *   tol       – convergence tolerance (max absolute change per sweep)
 *
 * Returns a PthreadsJacobiResult with iteration count, final error, and
 * a convergence flag.
 */
PthreadsJacobiResult jacobi_solve_pthreads(double **grid, int N,
                                           int num_threads,
                                           int max_iter, double tol);

#endif /* JACOBI_PTHREADS_H */
