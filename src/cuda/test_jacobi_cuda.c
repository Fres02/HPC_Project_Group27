#include "../core/jacobi_cuda.h"
#include "../core/grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Test 1: Verify convergence on small grid */
void test_cuda_convergence(void)
{
    printf("=== Test 1: CUDA Convergence ===\n");
    
    int N = 50;
    double **grid = alloc_grid(N);
    assert(grid != NULL);

    init_grid(grid, N);

    CudaJacobiResult result = jacobi_solve_cuda(grid, N, 10000, 1e-6);

    printf("CUDA Convergence Test:\n");
    printf("  Iterations: %d\n", result.iterations);
    printf("  Final Error: %g\n", result.final_error);
    printf("  Converged: %s\n", result.converged ? "YES" : "NO");

    assert(result.converged);
    assert(result.final_error < 1e-6);

    free_grid(grid, N);
    printf("✓ CUDA convergence test passed\n\n");
}

/* Test 2: Verify boundary conditions are preserved */
void test_cuda_boundary_preservation(void)
{
    printf("=== Test 2: CUDA Boundary Preservation ===\n");
    
    int N = 32;
    double **grid = alloc_grid(N);
    assert(grid != NULL);

    init_grid(grid, N);

    /* Save boundary values */
    double top_orig = grid[0][0];
    double bottom_orig = grid[N-1][0];
    double left_orig = grid[0][0];
    double right_orig = grid[0][N-1];

    CudaJacobiResult result = jacobi_solve_cuda(grid, N, 100, 1e-4);

    /* Verify boundaries are unchanged */
    assert(grid[0][0] == top_orig);          /* Top-left corner */
    assert(grid[N-1][0] == bottom_orig);     /* Bottom-left corner */
    assert(grid[0][N-1] == right_orig);      /* Top-right corner */
    assert(grid[N-1][N-1] == bottom_orig);   /* Bottom-right corner */

    /* Check top boundary */
    for (int j = 0; j < N; j++) {
        assert(fabs(grid[0][j] - BC_TOP) < 1e-10);
    }

    /* Check bottom boundary */
    for (int j = 0; j < N; j++) {
        assert(fabs(grid[N-1][j] - BC_BOTTOM) < 1e-10);
    }

    free_grid(grid, N);
    printf("✓ Boundary preservation test passed\n\n");
}

/* Test 3: Verify interior values are within valid range */
void test_cuda_solution_range(void)
{
    printf("=== Test 3: CUDA Solution Range ===\n");
    
    int N = 40;
    double **grid = alloc_grid(N);
    assert(grid != NULL);

    init_grid(grid, N);

    CudaJacobiResult result = jacobi_solve_cuda(grid, N, 5000, 1e-5);

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

/* Test 4: Verify CUDA handles different grid sizes */
void test_cuda_various_grid_sizes(void)
{
    printf("=== Test 4: CUDA Various Grid Sizes ===\n");
    
    int sizes[] = {32, 64, 128};
    
    for (int s = 0; s < 3; s++) {
        int N = sizes[s];
        double **grid = alloc_grid(N);
        assert(grid != NULL);

        init_grid(grid, N);

        printf("  Testing N=%d...", N);
        fflush(stdout);

        CudaJacobiResult result = jacobi_solve_cuda(grid, N, 1000, 1e-4);

        printf(" %d iterations\n", result.iterations);
        assert(result.final_error < 1e-4 || result.iterations < 1000);

        free_grid(grid, N);
    }

    printf("✓ Various grid sizes test passed\n\n");
}

/* Test 5: Verify error decreases monotonically (mostly) */
void test_cuda_error_monotonicity(void)
{
    printf("=== Test 5: CUDA Error Trend ===\n");
    
    int N = 48;
    double **grid = alloc_grid(N);
    assert(grid != NULL);

    init_grid(grid, N);

    printf("  Initial grid set up\n");
    printf("  Running 20 iterations with manual check...\n");

    /* Manually run a few iterations to check error trend */
    double prev_error = 1e6;
    int error_decreased = 0;

    for (int iter = 0; iter < 20; iter++) {
        CudaJacobiResult result = jacobi_solve_cuda(grid, N, 1, 0.0);
        
        if (iter > 0 && result.final_error < prev_error) {
            error_decreased++;
        }
        
        if (iter % 5 == 0) {
            printf("    Iter %2d: error = %g\n", iter, result.final_error);
        }
        
        prev_error = result.final_error;
    }

    printf("  Error decreased in %d out of 19 steps\n", error_decreased);
    assert(error_decreased >= 15);  /* Most steps should see error decrease */

    free_grid(grid, N);
    printf("✓ Error monotonicity test passed\n\n");
}

int main(void)
{
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║    CUDA Jacobi Solver Test Suite       ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    test_cuda_convergence();
    test_cuda_boundary_preservation();
    test_cuda_solution_range();
    test_cuda_various_grid_sizes();
    test_cuda_error_monotonicity();

    printf("╔════════════════════════════════════════╗\n");
    printf("║      All CUDA tests passed! ✓          ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    return 0;
}
