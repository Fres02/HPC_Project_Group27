#ifndef JACOBI_HYBRID_H
#define JACOBI_HYBRID_H

#include <mpi.h>

#include "../core/grid.h"
#include "../core/jacobi_serial.h"  /* for JacobiResult struct */

/*
 * Hybrid MPI + OpenMP version of the Jacobi solver.
 *
 * Two-level parallelism:
 *   - MPI: 1D row-wise domain decomposition across ranks (between nodes).
 *          Halo exchange with MPI_Sendrecv each iteration, global error
 *          via MPI_Allreduce(MPI_MAX).
 *   - OpenMP: each rank parallelizes its local stencil sweep using
 *          `#pragma omp parallel for` with a max-reduction.
 *
 * The same scatter/gather and ghost-row layout as the pure MPI version is
 * used; OpenMP is layered inside each rank's inner loops.
 *
 * On entry, rank 0 supplies the initial grid in `grid` (with Dirichlet
 * boundaries already applied). On exit, the converged solution is gathered
 * back into rank 0's `grid`. On other ranks `grid` is ignored.
 *
 * Parameters:
 *   grid        - initialized N x N grid on rank 0 (may be NULL elsewhere)
 *   N           - global grid dimension
 *   max_iter    - iteration cap
 *   tol         - convergence tolerance (max absolute change per sweep)
 *   num_threads - OpenMP threads per rank (0 = use OpenMP default)
 *   comm        - MPI communicator (e.g. MPI_COMM_WORLD)
 *
 * Returns: JacobiResult on every rank with iteration count, final
 * (global) error, and convergence flag.
 */
JacobiResult jacobi_solve_hybrid(double **grid, int N, int max_iter, double tol,
                                 int num_threads, MPI_Comm comm);

#endif /* JACOBI_HYBRID_H */
