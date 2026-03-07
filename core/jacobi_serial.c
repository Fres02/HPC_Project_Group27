#include "jacobi_serial.h"

#include <math.h>
#include <stdio.h>

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
    copy_grid(buf_b, grid, N);

    /* curr  – buffer we READ from each sweep
       next  – buffer we WRITE into each sweep
       We swap the row-pointer arrays (O(1)) instead of copying data (O(N²)). */
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

        swap_buffers(&curr, &next);

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
        copy_grid(grid, curr, N);

    free_grid(buf_b, N);
    return result;
}
