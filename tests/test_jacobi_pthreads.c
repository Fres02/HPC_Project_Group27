#include "../core/jacobi_pthreads.h"
#include "../core/grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Test 1: Verify convergence with single thread */
void test_pthreads_convergence_single_thread(void)
{
    printf("=== Test 1: Pthreads Convergence (1 thread) ===\n");
    
    int N = 50;
    double **grid = alloc_grid(N);
    assert(grid != NULL);

    init_grid(grid, N);

    PthreadsJacobiResult result = jacobi_solve_pthreads(grid, N, 1, 10000, 1e-6);

    printf("Pthreads (1 thread) Convergence Test:\n");
    printf("  Iterations: %d\n", result.iterations);
    printf("  Final Error: %g\n", result.final_error);
    printf("  Converged: %s\n", result.converged ? "YES" : "NO");

    assert(result.converged);
    assert(result.final_error < 1e-6);

    free_grid(grid, N);
    printf("✓ Pthreads (1 thread) convergence test passed\n\n");
}

/* Test 2: Verify convergence with multiple threads */
void test_pthreads_convergence_multi_thread(void)
{
    printf("=== Test 2: Pthreads Convergence (4 threads) ===\n");
    
    int N = 50;
    double **grid = alloc_grid(N);
    assert(grid != NULL);

    init_grid(grid, N);

    PthreadsJacobiResult result = jacobi_solve_pthreads(grid, N, 4, 10000, 1e-6);

    printf("Pthreads (4 threads) Convergence Test:\n");
    printf("  Iterations: %d\n", result.iterations);
    printf("  Final Error: %g\n", result.final_error);
    printf("  Converged: %s\n", result.converged ? "YES" : "NO");

    assert(result.converged);
    assert(result.final_error < 1e-6);

    free_grid(grid, N);
    printf("✓ Pthreads (4 threads) convergence test passed\n\n");
}

/* Test 3: Verify boundary conditions are preserved */
void test_pthreads_boundary_preservation(void)
{
    printf("=== Test 3: Pthreads Boundary Preservation ===\n");
    
    int N = 32;
    double **grid = alloc_grid(N);
    assert(grid != NULL);

    init_grid(grid, N);

    jacobi_solve_pthreads(grid, N, 2, 100, 1e-4);

    /* Check boundaries */
    for (int j = 0; j < N; j++) {
        assert(fabs(grid[0][j] - BC_TOP) < 1e-10);
        assert(fabs(grid[N-1][j] - BC_BOTTOM) < 1e-10);
        assert(fabs(grid[j][0] - BC_LEFT) < 1e-10);
        assert(fabs(grid[j][N-1] - BC_RIGHT) < 1e-10);
    }

    free_grid(grid, N);
    printf("✓ Boundary preservation test passed\n\n");
}

/* Test 4: Verify interior values are within valid range */
void test_pthreads_solution_range(void)
{
    printf("=== Test 4: Pthreads Solution Range ===\n");
    
    int N = 40;
    double **grid = alloc_grid(N);
    assert(grid != NULL);

    init_grid(grid, N);

    jacobi_solve_pthreads(grid, N, 3, 5000, 1e-5);

    /* Interior values should be between boundaries */
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            assert(grid[i][j] >= BC_BOTTOM - 1e-10);
            assert(grid[i][j] <= BC_TOP + 1e-10);
        }
    }

    free_grid(grid, N);
    printf("✓ Solution range test passed\n\n");
}

/* Test 5: Compare results across different thread counts */
void test_pthreads_thread_scalability(void)
{
    printf("=== Test 5: Pthreads Thread Scalability ===\n");
    
    int N = 60;
    int thread_counts[] = {1, 2, 4, 8};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    PthreadsJacobiResult results[num_configs];

    for (int c = 0; c < num_configs; c++) {
        int num_threads = thread_counts[c];
        double **grid = alloc_grid(N);
        assert(grid != NULL);

        init_grid(grid, N);

        printf("  Testing with %d thread(s)...", num_threads);
        fflush(stdout);

        results[c] = jacobi_solve_pthreads(grid, N, num_threads, 10000, 1e-6);

        printf(" %d iterations\n", results[c].iterations);

        free_grid(grid, N);
    }

    /* Verify all configurations converge to same tolerance */
    for (int c = 0; c < num_configs; c++) {
        assert(results[c].converged);
        assert(results[c].final_error < 1e-6);
    }

    /* Verify iteration counts are similar (should be identical for same algorithm) */
    int first_iter = results[0].iterations;
    for (int c = 1; c < num_configs; c++) {
        assert(results[c].iterations == first_iter);
    }

    printf("✓ Thread scalability test passed\n\n");
}

/* Test 6: Verify thread safety and data correctness */
void test_pthreads_data_integrity(void)
{
    printf("=== Test 6: Pthreads Data Integrity ===\n");
    
    int N = 44;
    double **grid1 = alloc_grid(N);
    double **grid2 = alloc_grid(N);
    assert(grid1 != NULL && grid2 != NULL);

    init_grid(grid1, N);
    init_grid(grid2, N);

    /* Solve with 1 thread (baseline) */
    PthreadsJacobiResult result1 = jacobi_solve_pthreads(grid1, N, 1, 1000, 1e-5);

    /* Solve with 4 threads */
    PthreadsJacobiResult result2 = jacobi_solve_pthreads(grid2, N, 4, 1000, 1e-5);

    /* Results should be numerically identical (same algorithm) */
    assert(result1.iterations == result2.iterations);
    assert(fabs(result1.final_error - result2.final_error) < 1e-15);

    /* Grid values should match */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(fabs(grid1[i][j] - grid2[i][j]) < 1e-15);
        }
    }

    free_grid(grid1, N);
    free_grid(grid2, N);
    printf("✓ Data integrity test passed\n\n");
}

int main(void)
{
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║  POSIX Threads Jacobi Solver Test Suite║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    test_pthreads_convergence_single_thread();
    test_pthreads_convergence_multi_thread();
    test_pthreads_boundary_preservation();
    test_pthreads_solution_range();
    test_pthreads_thread_scalability();
    test_pthreads_data_integrity();

    printf("╔════════════════════════════════════════╗\n");
    printf("║   All Pthreads tests passed! ✓         ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    return 0;
}
