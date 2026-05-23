/*
 * Demo application: POSIX Threads Jacobi Solver
 * 
 * This demo shows how to use the multi-threaded Jacobi solver
 * and demonstrates scaling across different thread counts.
 */

#include "core/jacobi_pthreads.h"
#include "core/grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║   POSIX Threads Jacobi Solver Demo     ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    int N = 200;
    int max_iter = 10000;
    double tol = 1e-6;

    printf("Configuration:\n");
    printf("  Grid Dimension:  %d × %d\n", N, N);
    printf("  Max Iterations:  %d\n", max_iter);
    printf("  Tolerance:       %g\n\n", tol);

    /* Test with different thread counts */
    int thread_counts[] = {1, 2, 4, 8};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("Thread Count Scaling Analysis:\n");
    printf("────────────────────────────────────────\n");
    printf("Threads │ Iterations │ Time (ms) │ Per-Iter (µs)\n");
    printf("────────┼────────────┼───────────┼──────────────\n");

    for (int t = 0; t < num_tests; t++) {
        int num_threads = thread_counts[t];

        double **grid = alloc_grid(N);
        if (!grid) {
            fprintf(stderr, "Failed to allocate grid\n");
            return 1;
        }

        init_grid(grid, N);

        clock_t start = clock();
        PthreadsJacobiResult result = jacobi_solve_pthreads(grid, N, num_threads, max_iter, tol);
        clock_t end = clock();

        double elapsed = (double)(end - start) / CLOCKS_PER_SEC * 1000;  // Convert to ms
        double per_iter = elapsed / result.iterations;

        printf("  %d    │    %5d   │   %.2f   │    %.2f\n",
               num_threads, result.iterations, elapsed, per_iter);

        free_grid(grid, N);
    }

    printf("────────────────────────────────────────\n\n");

    /* Detailed test with 4 threads */
    printf("Detailed Results (4 threads):\n");
    printf("────────────────────────────────────────\n");

    double **grid = alloc_grid(N);
    if (!grid) {
        fprintf(stderr, "Failed to allocate grid\n");
        return 1;
    }

    init_grid(grid, N);

    printf("Boundary conditions:\n");
    printf("  Top:    %g\n", BC_TOP);
    printf("  Bottom: %g\n", BC_BOTTOM);
    printf("  Left:   %g\n", BC_LEFT);
    printf("  Right:  %g\n\n", BC_RIGHT);

    clock_t start = clock();
    PthreadsJacobiResult result = jacobi_solve_pthreads(grid, N, 4, max_iter, tol);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Results:\n");
    printf("  Iterations:  %d\n", result.iterations);
    printf("  Final Error: %g\n", result.final_error);
    printf("  Converged:   %s\n", result.converged ? "YES" : "NO");
    printf("  CPU Time:    %.4f seconds\n", elapsed);
    printf("  Avg per iter: %.6f seconds\n\n", elapsed / result.iterations);

    /* Sample solution values */
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

    free_grid(grid, N);
    printf("\nDemo completed successfully!\n\n");

    return 0;
}
