#ifndef JACOBI_SERIAL_H
#define JACOBI_SERIAL_H

#include "grid.h"   /* grid allocation, init, BC constants, swap/copy */

/* Default solver parameters */
#define DEFAULT_N        100
#define DEFAULT_MAX_ITER 10000
#define DEFAULT_TOL      1e-6

/* Result struct returned by the solver */
typedef struct {
    int    iterations;   /* number of iterations performed */
    double final_error;  /* residual error at termination  */
    int    converged;    /* 1 if tolerance was met, 0 if max_iter hit */
} JacobiResult;

/*
 * Run the Jacobi iteration.
 *
 *   grid     – initialised N×N grid (boundary already set)
 *   N        – grid dimension
 *   max_iter – iteration cap
 *   tol      – convergence tolerance (max absolute change per sweep)
 *
 * Returns a JacobiResult with iteration count, final error, and
 * a convergence flag.
 */
JacobiResult jacobi_solve(double **grid, int N, int max_iter, double tol);

#endif /* JACOBI_SERIAL_H */
