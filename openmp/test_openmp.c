/*
 * Simple test program for OpenMP Jacobi solver
 * Quick validation without extensive benchmarking
 */

#include "../core/grid.h"
#include "../core/jacobi_serial.h"
#include "../core/convergence.h"
#include "jacobi_openmp.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv)
{
    int    N        = (argc > 1) ? atoi(argv[1]) : 100;
    int    max_iter = (argc > 2) ? atoi(argv[2]) : 10000;
    double tol      = (argc > 3) ? atof(argv[3]) : 1e-6;
    int    num_threads = (argc > 4) ? atoi(argv[4]) : 0;
    
    printf("==========================================\n");
    printf("  OpenMP Jacobi Solver - Quick Test\n");
    printf("==========================================\n");
    printf("  Grid Size  : %d × %d\n", N, N);
    printf("  Tolerance  : %.2e\n", tol);
    printf("  Max Iters  : %d\n", max_iter);
    printf("  Threads    : %d (0=default)\n", num_threads);
    printf("==========================================\n\n");
    
    /* Run serial baseline */
    printf("Running serial solver...\n");
    double **serial_grid = alloc_grid(N);
    if (!serial_grid) {
        fprintf(stderr, "Failed to allocate serial grid\n");
        return 1;
    }
    init_grid(serial_grid, N);
    
    double t_start = omp_get_wtime();
    JacobiResult serial_result = jacobi_solve(serial_grid, N, max_iter, tol);
    double t_serial = omp_get_wtime() - t_start;
    
    printf("  Iterations: %d\n", serial_result.iterations);
    printf("  Final Error: %.10e\n", serial_result.final_error);
    printf("  Converged: %s\n", serial_result.converged ? "YES" : "NO");
    printf("  Time: %.6f s\n\n", t_serial);
    
    /* Run OpenMP version */
    printf("Running OpenMP solver...\n");
    double **openmp_grid = alloc_grid(N);
    if (!openmp_grid) {
        fprintf(stderr, "Failed to allocate OpenMP grid\n");
        free_grid(serial_grid, N);
        return 1;
    }
    init_grid(openmp_grid, N);
    
    t_start = omp_get_wtime();
    JacobiResult openmp_result = jacobi_solve_openmp(openmp_grid, N, max_iter, tol, num_threads);
    double t_openmp = omp_get_wtime() - t_start;
    
    int actual_threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    
    printf("  Iterations: %d\n", openmp_result.iterations);
    printf("  Final Error: %.10e\n", openmp_result.final_error);
    printf("  Converged: %s\n", openmp_result.converged ? "YES" : "NO");
    printf("  Time: %.6f s\n", t_openmp);
    printf("  Threads Used: %d\n\n", actual_threads);
    
    /* Validate */
    printf("Validating...\n");
    ValidationResult val = validate_against_baseline(openmp_grid, serial_grid, N, tol);
    
    printf("  RMSE: %.10e\n", val.rmse);
    printf("  Max Error: %.10e\n", val.max_error);
    printf("  Status: %s\n\n", val.grids_match ? "PASS" : "FAIL");
    
    /* Performance */
    if (t_openmp > 0) {
        double speedup = t_serial / t_openmp;
        double efficiency = (speedup / actual_threads) * 100.0;
        printf("Performance:\n");
        printf("  Speedup: %.2fx\n", speedup);
        printf("  Efficiency: %.2f%%\n\n", efficiency);
    }
    
    /* Final result */
    if (val.grids_match) {
        printf("✓ TEST PASSED - OpenMP implementation is correct!\n");
    } else {
        printf("✗ TEST FAILED - Results do not match!\n");
    }
    
    printf("==========================================\n");
    
    free_grid(serial_grid, N);
    free_grid(openmp_grid, N);
    
    return val.grids_match ? 0 : 1;
}
