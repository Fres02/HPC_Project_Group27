#ifndef JACOBI_MPI_H
#define JACOBI_MPI_H

#include <mpi.h>

#include "../core/grid.h"
#include "../core/jacobi_serial.h"  /* for JacobiResult struct */

/*
 * MPI distributed-memory version of the Jacobi solver.
 *
 * Domain decomposition: 1D row-wise. The global N x N grid is split into
 * row slabs; each rank owns a slab of interior rows plus two ghost rows
 * (one above and one below) for halo exchange.
 *
 * Halo exchange: MPI_Sendrecv between adjacent ranks each iteration.
 * Global convergence check: MPI_Allreduce with MPI_MAX over the per-rank
 * maximum cell change.
 *
 * On entry, rank 0 supplies the initial grid in `grid` (with Dirichlet
 * boundaries already applied by init_grid). On exit, the converged
 * solution is gathered back into rank 0's `grid`. On other ranks
 * `grid` is ignored and may be NULL.
 *
 * Parameters:
 *   grid     - initialized N x N grid on rank 0 (may be NULL elsewhere)
 *   N        - global grid dimension
 *   max_iter - iteration cap
 *   tol      - convergence tolerance (max absolute change per sweep)
 *   comm     - MPI communicator (e.g. MPI_COMM_WORLD)
 *
 * Returns: JacobiResult on every rank with iteration count, final
 * (global) error, and convergence flag.
 */
JacobiResult jacobi_solve_mpi(double **grid, int N, int max_iter, double tol, MPI_Comm comm);

#endif /* JACOBI_MPI_H */
