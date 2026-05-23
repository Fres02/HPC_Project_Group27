/*
 * Test program for the Convergence and Validation module
 *
 * This program demonstrates:
 *   1. Running the serial Jacobi solver
 *   2. Printing convergence information
 *   3. Validating results against a baseline
 *   4. Computing various error metrics
 */

#include "../core/grid.h"
#include "../core/jacobi_serial.h"
#include "../core/convergence.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    /* Default parameters */
    int N = 100;
    int max_iter = 10000;
    double tol = 1e-6;

    /* Parse command-line arguments if provided */
    if (argc > 1)
        N = atoi(argv[1]);
    if (argc > 2)
        max_iter = atoi(argv[2]);
    if (argc > 3)
        tol = atof(argv[3]);

    printf("==========================================\n");
    printf("  Convergence & Validation Module Test\n");
    printf("==========================================\n");
    printf("  Grid Size      : %d × %d\n", N, N);
    printf("  Max Iterations : %d\n", max_iter);
    printf("  Tolerance      : %.2e\n", tol);
    printf("==========================================\n\n");

    /* ---------- Test 1: Serial Baseline Solver ---------- */

    printf("[Test 1] Running Serial Baseline Solver...\n");

    double **baseline_grid = alloc_grid(N);
    if (!baseline_grid)
    {
        fprintf(stderr, "Failed to allocate baseline grid\n");
        return 1;
    }

    init_grid(baseline_grid, N);

    JacobiResult baseline_result = jacobi_solve(baseline_grid, N, max_iter, tol);

    print_convergence_info(baseline_result, N, tol);

    /* ---------- Test 2: Second Run for Validation ---------- */

    printf("[Test 2] Running Second Solver (for validation demo)...\n");

    double **test_grid = alloc_grid(N);
    if (!test_grid)
    {
        fprintf(stderr, "Failed to allocate test grid\n");
        free_grid(baseline_grid, N);
        return 1;
    }

    init_grid(test_grid, N);

    JacobiResult test_result = jacobi_solve(test_grid, N, max_iter, tol);

    print_convergence_info(test_result, N, tol);

    /* ---------- Test 3: Validate Against Baseline ---------- */

    printf("[Test 3] Validating Test Grid Against Baseline...\n");

    ValidationResult val_result = validate_against_baseline(
        test_grid, baseline_grid, N, tol);

    print_validation_result(val_result);

    /* ---------- Test 4: Comparison Summary ---------- */

    printf("[Test 4] Comparison Summary...\n");

    print_comparison_summary(baseline_result, test_result, "Test");

    /* ---------- Test 5: Boundary Validation ---------- */

    printf("[Test 5] Boundary Validation...\n");

    int boundaries_ok = validate_boundaries(test_grid, baseline_grid, N);

    printf("  Boundary Check: %s\n\n", boundaries_ok ? "PASS" : "FAIL");

    /* ---------- Test 6: Individual Error Metrics ---------- */

    printf("[Test 6] Individual Error Metrics...\n");
    printf("------------------------------------------\n");

    double max_err = compute_max_error(test_grid, baseline_grid, N);
    double rmse = compute_rmse(test_grid, baseline_grid, N);
    double mae = compute_mean_absolute_error(test_grid, baseline_grid, N);
    double l2_norm = compute_l2_norm(test_grid, baseline_grid, N);

    printf("  Max Error : %.10e\n", max_err);
    printf("  RMSE      : %.10e\n", rmse);
    printf("  MAE       : %.10e\n", mae);
    printf("  L2 Norm   : %.10e\n", l2_norm);
    printf("------------------------------------------\n\n");

    /* ---------- Test 7: Convergence Checking ---------- */

    printf("[Test 7] Convergence Checking...\n");
    printf("------------------------------------------\n");

    int conv_check = check_convergence(baseline_result.final_error, tol);
    int is_conv = is_converged(baseline_result);

    printf("  check_convergence(): %s\n", conv_check ? "TRUE" : "FALSE");
    printf("  is_converged():      %s\n", is_conv ? "TRUE" : "FALSE");
    printf("------------------------------------------\n\n");

    /* Cleanup */
    free_grid(baseline_grid, N);
    free_grid(test_grid, N);

    printf("==========================================\n");
    printf("  All Tests Completed Successfully!\n");
    printf("==========================================\n");

    return 0;
}
