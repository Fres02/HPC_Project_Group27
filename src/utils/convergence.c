#include "convergence.h"

#include <math.h>
#include <stdio.h>

/* --------------------------------------------------------------------------
 * Error Computation Functions
 * -------------------------------------------------------------------------- */

double compute_max_error(double **grid1, double **grid2, int N)
{
    double max_err = 0.0;

    /* Compare interior points only (rows 1..N-2, cols 1..N-2) */
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            double diff = fabs(grid1[i][j] - grid2[i][j]);
            if (diff > max_err)
            {
                max_err = diff;
            }
        }
    }

    return max_err;
}

double compute_rmse(double **grid1, double **grid2, int N)
{
    double sum_sq = 0.0;
    int num_points = 0;

    /* Sum squared differences for interior points */
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            double diff = grid1[i][j] - grid2[i][j];
            sum_sq += diff * diff;
            num_points++;
        }
    }

    /* Avoid division by zero for very small grids */
    if (num_points == 0)
    {
        return 0.0;
    }

    return sqrt(sum_sq / num_points);
}

double compute_mean_absolute_error(double **grid1, double **grid2, int N)
{
    double sum_abs = 0.0;
    int num_points = 0;

    /* Sum absolute differences for interior points */
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            sum_abs += fabs(grid1[i][j] - grid2[i][j]);
            num_points++;
        }
    }

    /* Avoid division by zero */
    if (num_points == 0)
    {
        return 0.0;
    }

    return sum_abs / num_points;
}

double compute_l2_norm(double **grid1, double **grid2, int N)
{
    double sum_sq = 0.0;

    /* Sum squared differences for interior points */
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            double diff = grid1[i][j] - grid2[i][j];
            sum_sq += diff * diff;
        }
    }

    return sqrt(sum_sq);
}

/* --------------------------------------------------------------------------
 * Convergence Testing
 * -------------------------------------------------------------------------- */

int check_convergence(double error, double tolerance)
{
    return (error < tolerance) ? 1 : 0;
}

int is_converged(JacobiResult result)
{
    return result.converged;
}

/* --------------------------------------------------------------------------
 * Validation Against Baseline
 * -------------------------------------------------------------------------- */

ValidationResult validate_against_baseline(double **test_grid,
                                           double **baseline_grid,
                                           int N,
                                           double tolerance)
{
    ValidationResult result;

    /* Compute all error metrics */
    result.max_error = compute_max_error(test_grid, baseline_grid, N);
    result.rmse = compute_rmse(test_grid, baseline_grid, N);
    result.mae = compute_mean_absolute_error(test_grid, baseline_grid, N);

    /* Check if grids match within tolerance (using max error as criterion) */
    result.grids_match = (result.max_error < tolerance) ? 1 : 0;

    return result;
}

/* --------------------------------------------------------------------------
 * Boundary Validation
 * -------------------------------------------------------------------------- */

int validate_boundaries(double **grid1, double **grid2, int N)
{
    /* Check top row */
    for (int j = 0; j < N; j++)
    {
        if (grid1[0][j] != grid2[0][j])
        {
            return 0;
        }
    }

    /* Check bottom row */
    for (int j = 0; j < N; j++)
    {
        if (grid1[N - 1][j] != grid2[N - 1][j])
        {
            return 0;
        }
    }

    /* Check left column (excluding corners already checked) */
    for (int i = 1; i < N - 1; i++)
    {
        if (grid1[i][0] != grid2[i][0])
        {
            return 0;
        }
    }

    /* Check right column (excluding corners already checked) */
    for (int i = 1; i < N - 1; i++)
    {
        if (grid1[i][N - 1] != grid2[i][N - 1])
        {
            return 0;
        }
    }

    return 1; /* All boundaries match */
}

/* --------------------------------------------------------------------------
 * Reporting and Output
 * -------------------------------------------------------------------------- */

void print_validation_result(ValidationResult result)
{
    printf("\n");
    printf("==========================================\n");
    printf("         VALIDATION RESULTS\n");
    printf("==========================================\n");
    printf("  Maximum Absolute Error : %.10e\n", result.max_error);
    printf("  RMSE                   : %.10e\n", result.rmse);
    printf("  Mean Absolute Error    : %.10e\n", result.mae);
    printf("------------------------------------------\n");

    if (result.grids_match)
    {
        printf("  Status: PASS - Grids match within tolerance\n");
    }
    else
    {
        printf("  Status: FAIL - Grids differ beyond tolerance\n");
    }

    printf("==========================================\n");
    printf("\n");
}

void print_convergence_info(JacobiResult result, int N, double tolerance)
{
    printf("\n");
    printf("==========================================\n");
    printf("        CONVERGENCE INFORMATION\n");
    printf("==========================================\n");
    printf("  Grid Size              : %d × %d\n", N, N);
    printf("  Total Grid Points      : %d\n", N * N);
    printf("  Interior Points        : %d\n", (N - 2) * (N - 2));
    printf("  Tolerance              : %.10e\n", tolerance);
    printf("------------------------------------------\n");
    printf("  Iterations Performed   : %d\n", result.iterations);
    printf("  Final Error            : %.10e\n", result.final_error);

    if (result.converged)
    {
        printf("  Convergence Status     : CONVERGED\n");
    }
    else
    {
        printf("  Convergence Status     : NOT CONVERGED (max iter reached)\n");
    }

    printf("==========================================\n");
    printf("\n");
}

void print_comparison_summary(JacobiResult serial_result,
                              JacobiResult parallel_result,
                              const char *label)
{
    printf("\n");
    printf("==========================================\n");
    printf("   SERIAL vs %s COMPARISON\n", label);
    printf("==========================================\n");

    /* Iterations comparison */
    printf("  Iterations:\n");
    printf("    Serial     : %d\n", serial_result.iterations);
    printf("    %-10s : %d\n", label, parallel_result.iterations);

    if (serial_result.iterations == parallel_result.iterations)
    {
        printf("    Match      : YES\n");
    }
    else
    {
        printf("    Match      : NO (diff = %d)\n",
               abs(parallel_result.iterations - serial_result.iterations));
    }

    printf("------------------------------------------\n");

    /* Final error comparison */
    printf("  Final Error:\n");
    printf("    Serial     : %.10e\n", serial_result.final_error);
    printf("    %-10s : %.10e\n", label, parallel_result.final_error);
    printf("    Difference : %.10e\n",
           fabs(parallel_result.final_error - serial_result.final_error));

    printf("------------------------------------------\n");

    /* Convergence status comparison */
    printf("  Convergence:\n");
    printf("    Serial     : %s\n", serial_result.converged ? "YES" : "NO");
    printf("    %-10s : %s\n", label, parallel_result.converged ? "YES" : "NO");

    if (serial_result.converged == parallel_result.converged)
    {
        printf("    Match      : YES\n");
    }
    else
    {
        printf("    Match      : NO\n");
    }

    printf("==========================================\n");
    printf("\n");
}
