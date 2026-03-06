#include "jacobi_serial.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Grid allocation / deallocation
 * -------------------------------------------------------------------------- */

double **alloc_grid(int N)
{
    /* Allocate a contiguous N×N block so pointer arithmetic is cache-friendly */
    double  *data = (double *)malloc((size_t)N * N * sizeof(double));
    double **rows = (double **)malloc((size_t)N * sizeof(double *));
    if (!data || !rows) {
        fprintf(stderr, "alloc_grid: out of memory (N=%d)\n", N);
        free(data);
        free(rows);
        return NULL;
    }
    for (int i = 0; i < N; i++)
        rows[i] = data + (size_t)i * N;
    return rows;
}

void free_grid(double **grid, int N)
{
    if (!grid) return;
    free(grid[0]); /* free the contiguous data block */
    free(grid);
    (void)N;       /* N not needed but kept for API symmetry */
}

/* --------------------------------------------------------------------------
 * Grid initialisation
 * -------------------------------------------------------------------------- */

void init_grid(double **grid, int N)
{
    /* Zero the entire grid (interior + boundary) */
    memset(grid[0], 0, (size_t)N * N * sizeof(double));

    /* Fixed Dirichlet boundary conditions.
     * Rows are set first for the full width (corners belong to top/bottom).
     * Columns cover only interior rows to avoid overwriting corner values. */
    for (int j = 0; j < N; j++) {
        grid[0][j]   = BC_TOP;    /* top row    (includes corners) */
        grid[N-1][j] = BC_BOTTOM; /* bottom row (includes corners) */
    }
    for (int i = 1; i < N - 1; i++) {
        grid[i][0]   = BC_LEFT;   /* left column  (interior rows only) */
        grid[i][N-1] = BC_RIGHT;  /* right column (interior rows only) */
    }
}

/* --------------------------------------------------------------------------
 * Jacobi solver
 * -------------------------------------------------------------------------- */

JacobiResult jacobi_solve(double **grid, int N, int max_iter, double tol)
{
    JacobiResult result = {0, 0.0, 0};

    /* Allocate second buffer for double-buffering; copy grid into it so
       boundary values are present in both buffers from the start.        */
    double **buf_b = alloc_grid(N);
    if (!buf_b) {
        fprintf(stderr, "jacobi_solve: failed to allocate buffer\n");
        return result;
    }
    memcpy(buf_b[0], grid[0], (size_t)N * N * sizeof(double));

    /* curr  – buffer we READ from each sweep
       next  – buffer we WRITE into each sweep
       We swap the row-pointer arrays (O(N)) instead of copying data (O(N²)). */
    double **curr = grid;
    double **next = buf_b;

    for (int iter = 0; iter < max_iter; iter++) {

        double max_err = 0.0;

        /* Sweep over interior points only */
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                next[i][j] = 0.25 * (curr[i-1][j] +
                                     curr[i+1][j] +
                                     curr[i][j-1] +
                                     curr[i][j+1]);

                double diff = fabs(next[i][j] - curr[i][j]);
                if (diff > max_err)
                    max_err = diff;
            }
        }

        /* Swap buffer pointers – O(N), no data movement */
        double **tmp = curr;
        curr = next;
        next = tmp;

        result.iterations  = iter + 1;
        result.final_error = max_err;

        if (max_err < tol) {
            result.converged = 1;
            break;
        }
    }

    /* Ensure the result always lives in `grid` (the caller's buffer).
       If an odd number of swaps occurred, curr points to buf_b. */
    if (curr != grid)
        memcpy(grid[0], curr[0], (size_t)N * N * sizeof(double));

    free_grid(buf_b, N);
    return result;
}
