/*
 * OpenMP Jacobi Solver - Benchmark and Validation Program
 * 
 * This program:
 *   1. Runs the serial baseline solver
 *   2. Runs the OpenMP parallel solver with different thread counts
 *   3. Validates correctness using the convergence module
 *   4. Measures performance (timing, speedup, efficiency)
 *   5. Generates data for the analysis report
 */

#include "../core/grid.h"
#include "../core/jacobi_serial.h"
#include "../core/convergence.h"
#include "jacobi_openmp.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void print_header(const char *title) {
    printf("\n");
    printf("==================================================\n");
    printf("  %s\n", title);
    printf("==================================================\n");
}

void print_separator() {
    printf("--------------------------------------------------\n");
}

int main(int argc, char **argv)
{
    /* Default parameters */
    int    N        = 100;
    int    max_iter = 10000;
    double tol      = 1e-6;
    int    max_threads = omp_get_max_threads();
    
    /* Parse command-line arguments */
    if (argc > 1) N        = atoi(argv[1]);
    if (argc > 2) max_iter = atoi(argv[2]);
    if (argc > 3) tol      = atof(argv[3]);
    if (argc > 4) max_threads = atoi(argv[4]);
    
    print_header("OpenMP Jacobi Solver - Benchmark & Validation");
    
    printf("Configuration:\n");
    printf("  Grid Size       : %d × %d\n", N, N);
    printf("  Interior Points : %d\n", (N-2) * (N-2));
    printf("  Max Iterations  : %d\n", max_iter);
    printf("  Tolerance       : %.2e\n", tol);
    printf("  Max Threads     : %d\n", max_threads);
    printf("\n");
    
    /* ========================================
     * STEP 1: Run Serial Baseline
     * ======================================== */
    
    print_header("Step 1: Serial Baseline Solver");
    
    double **serial_grid = alloc_grid(N);
    if (!serial_grid) {
        fprintf(stderr, "Error: Failed to allocate serial grid\n");
        return 1;
    }
    init_grid(serial_grid, N);
    
    double start_time = omp_get_wtime();
    JacobiResult serial_result = jacobi_solve(serial_grid, N, max_iter, tol);
    double end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    
    print_convergence_info(serial_result, N, tol);
    printf("  Execution Time  : %.6f seconds\n", serial_time);
    
    /* ========================================
     * STEP 2: Run OpenMP with Different Thread Counts
     * ======================================== */
    
    print_header("Step 2: OpenMP Parallel Solver - Thread Scaling");
    
    printf("%-10s %-12s %-12s %-15s %-10s %-10s %-10s\n",
           "Threads", "Time (s)", "Speedup", "Efficiency (%)", "Iters", "Error", "Valid");
    print_separator();
    
    /* Test with 1, 2, 4, 8, ... threads up to max_threads */
    int thread_counts[] = {1, 2, 4, 8, 16, 32};
    int num_tests = 6;
    
    for (int t = 0; t < num_tests; t++) {
        int num_threads = thread_counts[t];
        if (num_threads > max_threads) break;
        
        /* Allocate grid for this test */
        double **openmp_grid = alloc_grid(N);
        if (!openmp_grid) {
            fprintf(stderr, "Error: Failed to allocate OpenMP grid\n");
            free_grid(serial_grid, N);
            return 1;
        }
        init_grid(openmp_grid, N);
        
        /* Run OpenMP solver */
        start_time = omp_get_wtime();
        JacobiResult openmp_result = jacobi_solve_openmp(openmp_grid, N, max_iter, tol, num_threads);
        end_time = omp_get_wtime();
        double openmp_time = end_time - start_time;
        
        /* Validate against serial */
        ValidationResult val = validate_against_baseline(openmp_grid, serial_grid, N, tol);
        
        /* Calculate performance metrics */
        double speedup = serial_time / openmp_time;
        double efficiency = (speedup / num_threads) * 100.0;
        
        /* Print results */
        printf("%-10d %-12.6f %-12.2f %-15.2f %-10d %-10.2e %s\n",
               num_threads,
               openmp_time,
               speedup,
               efficiency,
               openmp_result.iterations,
               val.rmse,
               val.grids_match ? "PASS" : "FAIL");
        
        free_grid(openmp_grid, N);
    }
    
    print_separator();
    
    /* ========================================
     * STEP 3: Detailed Validation with Default Threads
     * ======================================== */
    
    print_header("Step 3: Detailed Validation");
    
    double **openmp_grid = alloc_grid(N);
    if (!openmp_grid) {
        fprintf(stderr, "Error: Failed to allocate OpenMP grid\n");
        free_grid(serial_grid, N);
        return 1;
    }
    init_grid(openmp_grid, N);
    
    printf("Running OpenMP solver with %d threads...\n", max_threads);
    start_time = omp_get_wtime();
    JacobiResult openmp_result = jacobi_solve_openmp(openmp_grid, N, max_iter, tol, 0);
    end_time = omp_get_wtime();
    double openmp_time = end_time - start_time;
    
    print_convergence_info(openmp_result, N, tol);
    printf("  Execution Time  : %.6f seconds\n", openmp_time);
    
    /* Validate correctness */
    ValidationResult validation = validate_against_baseline(openmp_grid, serial_grid, N, tol);
    print_validation_result(validation);
    
    /* Check boundaries */
    int boundaries_ok = validate_boundaries(openmp_grid, serial_grid, N);
    printf("Boundary Validation: %s\n", boundaries_ok ? "PASS" : "FAIL");
    
    /* ========================================
     * STEP 4: Comparison Summary
     * ======================================== */
    
    print_comparison_summary(serial_result, openmp_result, "OpenMP");
    
    /* ========================================
     * STEP 5: Performance Analysis
     * ======================================== */
    
    print_header("Step 5: Performance Analysis");
    
    double speedup = serial_time / openmp_time;
    double efficiency = (speedup / max_threads) * 100.0;
    
    printf("Timing Results:\n");
    printf("  Serial Time     : %.6f s\n", serial_time);
    printf("  OpenMP Time     : %.6f s (with %d threads)\n", openmp_time, max_threads);
    printf("  Speedup         : %.2fx\n", speedup);
    printf("  Efficiency      : %.2f%%\n", efficiency);
    printf("\n");
    
    /* ========================================
     * STEP 6: Analysis Report Data
     * ======================================== */
    
    print_header("Step 6: Data for Analysis Report");
    
    printf("Accuracy Metrics (Required for Deliverable 2):\n");
    printf("  RMSE                 : %.10e\n", validation.rmse);
    printf("  Maximum Error        : %.10e\n", validation.max_error);
    printf("  Mean Absolute Error  : %.10e\n", validation.mae);
    printf("  Validation Status    : %s\n", validation.grids_match ? "PASS" : "FAIL");
    printf("\n");
    
    printf("Convergence Comparison:\n");
    printf("  Serial Iterations    : %d\n", serial_result.iterations);
    printf("  OpenMP Iterations    : %d\n", openmp_result.iterations);
    printf("  Iterations Match     : %s\n", 
           (serial_result.iterations == openmp_result.iterations) ? "YES" : "NO");
    printf("  Serial Final Error   : %.10e\n", serial_result.final_error);
    printf("  OpenMP Final Error   : %.10e\n", openmp_result.final_error);
    printf("\n");
    
    printf("Performance Metrics:\n");
    printf("  Number of Threads    : %d\n", max_threads);
    printf("  Serial Time          : %.6f s\n", serial_time);
    printf("  Parallel Time        : %.6f s\n", openmp_time);
    printf("  Speedup              : %.2fx\n", speedup);
    printf("  Parallel Efficiency  : %.2f%%\n", efficiency);
    printf("\n");
    
    /* ========================================
     * STEP 7: Final Status
     * ======================================== */
    
    print_header("Final Status");
    
    if (validation.grids_match && boundaries_ok) {
        printf("  ✓ All validations PASSED\n");
        printf("  ✓ OpenMP implementation is correct\n");
        printf("  ✓ Results match serial baseline within tolerance\n");
        printf("  ✓ Speedup achieved: %.2fx with %d threads\n", speedup, max_threads);
    } else {
        printf("  ✗ Validation FAILED\n");
        if (!validation.grids_match) {
            printf("  ✗ Grids do not match within tolerance\n");
        }
        if (!boundaries_ok) {
            printf("  ✗ Boundaries not preserved correctly\n");
        }
    }
    
    printf("==================================================\n\n");
    
    /* Cleanup */
    free_grid(serial_grid, N);
    free_grid(openmp_grid, N);
    
    return validation.grids_match ? 0 : 1;
}
