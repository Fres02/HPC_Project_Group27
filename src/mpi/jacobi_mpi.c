#include "jacobi_mpi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Compute how many interior rows go to each rank using a balanced split.
 * Interior rows are global rows 1..N-2 (total N-2 rows). */
static void compute_distribution(int N, int size,
                                 int *counts,    /* interior rows per rank */
                                 int *starts)    /* global index of first interior row */
{
    int interior = N - 2;
    int base = interior / size;
    int extra = interior % size;
    int offset = 1; /* first interior row globally */
    for (int r = 0; r < size; r++) {
        counts[r] = base + (r < extra ? 1 : 0);
        starts[r] = offset;
        offset += counts[r];
    }
}

JacobiResult jacobi_solve_mpi(double **grid, int N, int max_iter, double tol, MPI_Comm comm)
{
    JacobiResult result = {0, 0.0, 0};

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Bail if the grid is too small to distribute */
    if (N - 2 < size) {
        if (rank == 0) {
            fprintf(stderr, "jacobi_solve_mpi: grid too small (N=%d) for %d ranks\n", N, size);
        }
        return result;
    }

    /* Per-rank row distribution */
    int *counts = (int *)malloc((size_t)size * sizeof(int));
    int *starts = (int *)malloc((size_t)size * sizeof(int));
    if (!counts || !starts) {
        fprintf(stderr, "jacobi_solve_mpi: distribution alloc failed on rank %d\n", rank);
        free(counts); free(starts);
        return result;
    }
    compute_distribution(N, size, counts, starts);
    int local_rows = counts[rank];          /* owned interior rows */
    int local_h    = local_rows + 2;        /* with ghost rows */

    /* Allocate local slab using the same row-pointer layout as core/grid.c.
     * Contiguous data block so we can MPI_Send/Recv full rows easily. */
    double  *local_data = (double *)malloc((size_t)local_h * N * sizeof(double));
    double **local      = (double **)malloc((size_t)local_h * sizeof(double *));
    if (!local_data || !local) {
        fprintf(stderr, "jacobi_solve_mpi: local buffer alloc failed on rank %d\n", rank);
        free(local_data); free(local); free(counts); free(starts);
        return result;
    }
    for (int i = 0; i < local_h; i++)
        local[i] = local_data + (size_t)i * N;

    /* -------- Scatter initial grid: each rank receives its slab plus ghost rows --------
     * Slab for rank r covers global rows [starts[r]-1 .. starts[r]+counts[r]] inclusive,
     * which is exactly (counts[r] + 2) rows. The top and bottom ghost rows on the boundary
     * ranks coincide with the Dirichlet boundary rows of the global grid. */
    int *send_counts = NULL;
    int *send_displs = NULL;
    if (rank == 0) {
        send_counts = (int *)malloc((size_t)size * sizeof(int));
        send_displs = (int *)malloc((size_t)size * sizeof(int));
        for (int r = 0; r < size; r++) {
            send_counts[r] = (counts[r] + 2) * N;
            send_displs[r] = (starts[r] - 1) * N;   /* include one row above */
        }
    }

    MPI_Scatterv(rank == 0 ? grid[0] : NULL,
                 send_counts, send_displs, MPI_DOUBLE,
                 local_data, local_h * N, MPI_DOUBLE,
                 0, comm);

    /* -------- Second buffer for double-buffering -------- */
    double  *next_data = (double *)malloc((size_t)local_h * N * sizeof(double));
    double **next      = (double **)malloc((size_t)local_h * sizeof(double *));
    if (!next_data || !next) {
        fprintf(stderr, "jacobi_solve_mpi: next buffer alloc failed on rank %d\n", rank);
        free(local_data); free(local);
        free(next_data); free(next);
        free(counts); free(starts);
        free(send_counts); free(send_displs);
        return result;
    }
    for (int i = 0; i < local_h; i++)
        next[i] = next_data + (size_t)i * N;
    /* Initialize next with a full copy so unmodified cells (boundaries, ghost rows)
     * retain correct values when we swap buffers. */
    memcpy(next_data, local_data, (size_t)local_h * N * sizeof(double));

    /* Neighbor ranks (MPI_PROC_NULL at ends makes Sendrecv a no-op there) */
    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    double **curr = local;
    double **nxt  = next;

    int iter;
    double global_max = 0.0;
    for (iter = 0; iter < max_iter; iter++) {

        /* Halo exchange: send first owned row up, receive top ghost from up;
         *                send last owned row down, receive bottom ghost from down. */
        MPI_Sendrecv(curr[1],            N, MPI_DOUBLE, up,   0,
                     curr[local_rows+1], N, MPI_DOUBLE, down, 0,
                     comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(curr[local_rows],   N, MPI_DOUBLE, down, 1,
                     curr[0],            N, MPI_DOUBLE, up,   1,
                     comm, MPI_STATUS_IGNORE);

        /* Jacobi sweep over owned interior cells */
        double local_max = 0.0;
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j < N - 1; j++) {
                double v = 0.25 * (curr[i-1][j] + curr[i+1][j] +
                                   curr[i][j-1] + curr[i][j+1]);
                double diff = fabs(v - curr[i][j]);
                if (diff > local_max) local_max = diff;
                nxt[i][j] = v;
            }
        }

        /* Pointer swap (O(1), and we keep boundary/ghost cells correct because
         * we initialized next as a full copy of local at the start). */
        double **tmp = curr;
        curr = nxt;
        nxt  = tmp;

        /* Global convergence check */
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

        if (global_max < tol) {
            iter++;
            break;
        }
    }

    result.iterations  = iter;
    result.final_error = global_max;
    result.converged   = (global_max < tol) ? 1 : 0;

    /* -------- Gather owned interior rows back to rank 0's full grid -------- */
    int *recv_counts = NULL;
    int *recv_displs = NULL;
    if (rank == 0) {
        recv_counts = (int *)malloc((size_t)size * sizeof(int));
        recv_displs = (int *)malloc((size_t)size * sizeof(int));
        for (int r = 0; r < size; r++) {
            recv_counts[r] = counts[r] * N;
            recv_displs[r] = starts[r] * N;
        }
    }

    /* Send rows 1..local_rows of curr (skipping top ghost) */
    MPI_Gatherv(curr[1], local_rows * N, MPI_DOUBLE,
                rank == 0 ? grid[0] : NULL,
                recv_counts, recv_displs, MPI_DOUBLE,
                0, comm);

    /* Cleanup */
    free(local_data); free(local);
    free(next_data);  free(next);
    free(counts);     free(starts);
    free(send_counts); free(send_displs);
    free(recv_counts); free(recv_displs);

    return result;
}
