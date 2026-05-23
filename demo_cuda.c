/*
 * Demo application: CUDA Jacobi Solver
 * 
 * This demo shows how to use the CUDA-accelerated Jacobi solver.
 * It initializes a grid and solves it on the GPU.
 */

#include "core/jacobi_cuda.h"
#include "core/grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║   CUDA Jacobi Solver Demo              ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    /* Grid configuration */
    int N = 256;
    int max_iter = 10000;
    double tol = 1e-6;

    printf("Grid Configuration:\n");
    printf("  Dimension:     %d × %d\n", N, N);
    printf("  Max Iterations: %d\n", max_iter);
    printf("  Tolerance:     %g\n\n", tol);

    /* Allocate and initialize grid */
    printf("Initializing grid...\n");
    double **grid = alloc_grid(N);
    if (!grid) {
        fprintf(stderr, "Failed to allocate grid\n");
        return 1;
    }

    init_grid(grid, N);
    printf("Grid initialized with default boundary conditions.\n");
    printf("  BC_TOP:    %g\n", BC_TOP);
    printf("  BC_BOTTOM: %g\n", BC_BOTTOM);
    printf("  BC_LEFT:   %g\n", BC_LEFT);
    printf("  BC_RIGHT:  %g\n\n", BC_RIGHT);

    /* Run CUDA solver */
    printf("Starting CUDA Jacobi solver...\n");
    printf("────────────────────────────────────────\n");

    clock_t start = clock();
    CudaJacobiResult result = jacobi_solve_cuda(grid, N, max_iter, tol);
    clock_t end = clock();

    printf("────────────────────────────────────────\n");
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nResults:\n");
    printf("  Iterations:    %d\n", result.iterations);
    printf("  Final Error:   %g\n", result.final_error);
    printf("  Converged:     %s\n", result.converged ? "YES" : "NO");
    printf("  CPU Time:      %.4f seconds\n", elapsed);
    printf("  Avg per iter:  %.6f seconds\n\n", elapsed / result.iterations);

    /* Sample the solution */
    printf("Solution Sample (interior points):\n");
    for (int i = 0; i < 3; i++) {
        int row = N / 4 + i * N / 8;
        printf("  Row %d: ", row);
        for (int j = 0; j < 3; j++) {
            int col = N / 4 + j * N / 8;
            printf("grid[%d][%d]=%.6f  ", row, col, grid[row][col]);
        }
        printf("\n");
    }

    /* Cleanup */
    free_grid(grid, N);
    printf("\nDemo completed successfully!\n\n");

    return 0;
}
