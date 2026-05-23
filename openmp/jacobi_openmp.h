#ifndef JACOBI_OPENMP_H
#define JACOBI_OPENMP_H

#include "../core/grid.h"
#include "../core/jacobi_serial.h"  /* for JacobiResult struct */

/*
 * OpenMP parallel version of the Jacobi solver.
 * 
 * Uses OpenMP directives to parallelize the grid update loops
 * and handle reductions for error calculation.
 * 
 * Parameters:
 *   grid     - initialized N×N grid (boundary already set)
 *   N        - grid dimension
 *   max_iter - iteration cap
 *   tol      - convergence tolerance
 *   num_threads - number of OpenMP threads to use (0 = default)
 * 
 * Returns: JacobiResult with iteration count, final error, and convergence flag
 */
JacobiResult jacobi_solve_openmp(double **grid, int N, int max_iter, double tol, int num_threads);

#endif /* JACOBI_OPENMP_H */
