#include "grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Allocation / deallocation
 * -------------------------------------------------------------------------- */

double **alloc_grid(int N)
{
    /* Single contiguous block: cache-friendly and frees in one call */
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
    (void)N;
}

/* --------------------------------------------------------------------------
 * Initialisation
 * -------------------------------------------------------------------------- */

void init_grid(double **grid, int N)
{
    init_grid_custom(grid, N, BC_TOP, BC_BOTTOM, BC_LEFT, BC_RIGHT, 0.0);
}

void init_grid_custom(double **grid, int N,
                      double top, double bottom,
                      double left, double right,
                      double interior_val)
{
    /* Fill the whole grid (including boundary) with interior_val first,
     * then overwrite the four edges.  This guarantees a fully defined grid
     * even when interior_val != 0. */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            grid[i][j] = interior_val;

    /* Top and bottom rows span the full width (corners included) */
    for (int j = 0; j < N; j++) {
        grid[0][j]   = top;
        grid[N-1][j] = bottom;
    }
    /* Left and right columns cover interior rows only to preserve corners */
    for (int i = 1; i < N - 1; i++) {
        grid[i][0]   = left;
        grid[i][N-1] = right;
    }
}

/* --------------------------------------------------------------------------
 * Interior fill
 * -------------------------------------------------------------------------- */

void fill_interior(double **grid, int N, double value)
{
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            grid[i][j] = value;
}

/* --------------------------------------------------------------------------
 * Buffer utilities
 * -------------------------------------------------------------------------- */

void swap_buffers(double ***a, double ***b)
{
    double **tmp = *a;
    *a = *b;
    *b = tmp;
}

void copy_grid(double **dst, double **src, int N)
{
    memcpy(dst[0], src[0], (size_t)N * N * sizeof(double));
}
