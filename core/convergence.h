#ifndef CONVERGENCE_H
#define CONVERGENCE_H

#include "jacobi_serial.h" /* for JacobiResult struct */

/* --------------------------------------------------------------------------
 * Convergence and Validation Module
 *
 * This module provides:
 *   - Error computation metrics (max error, RMSE, MAE)
 *   - Convergence testing utilities
 *   - Validation against serial baseline
 *   - Numerical tolerance validation
 * -------------------------------------------------------------------------- */

/**
 * Validation result structure containing multiple error metrics
 * for comparing a test grid against a baseline (reference) grid.
 */
typedef struct
{
    double max_error; /* Maximum absolute difference */
    double rmse;      /* Root Mean Square Error */
    double mae;       /* Mean Absolute Error */
    int grids_match;  /* 1 if error < tolerance, 0 otherwise */
} ValidationResult;

/* --------------------------------------------------------------------------
 * Error Computation Functions
 * -------------------------------------------------------------------------- */

/**
 * Compute the maximum absolute difference between two grids.
 *
 * Compares all interior points (rows 1..N-2, cols 1..N-2).
 *
 * @param grid1  First grid (N×N)
 * @param grid2  Second grid (N×N)
 * @param N      Grid dimension
 * @return       Maximum absolute error
 */
double compute_max_error(double **grid1, double **grid2, int N);

/**
 * Compute Root Mean Square Error (RMSE) between two grids.
 *
 * RMSE = sqrt( sum((grid1[i][j] - grid2[i][j])^2) / num_points )
 *
 * Compares only interior points.
 *
 * @param grid1  First grid (N×N)
 * @param grid2  Second grid (N×N)
 * @param N      Grid dimension
 * @return       RMSE value
 */
double compute_rmse(double **grid1, double **grid2, int N);

/**
 * Compute Mean Absolute Error (MAE) between two grids.
 *
 * MAE = sum(|grid1[i][j] - grid2[i][j]|) / num_points
 *
 * Compares only interior points.
 *
 * @param grid1  First grid (N×N)
 * @param grid2  Second grid (N×N)
 * @param N      Grid dimension
 * @return       MAE value
 */
double compute_mean_absolute_error(double **grid1, double **grid2, int N);

/**
 * Compute the L2 norm (Euclidean norm) of the difference between two grids.
 *
 * L2 = sqrt( sum((grid1[i][j] - grid2[i][j])^2) )
 *
 * Compares only interior points.
 *
 * @param grid1  First grid (N×N)
 * @param grid2  Second grid (N×N)
 * @param N      Grid dimension
 * @return       L2 norm value
 */
double compute_l2_norm(double **grid1, double **grid2, int N);

/* --------------------------------------------------------------------------
 * Convergence Testing
 * -------------------------------------------------------------------------- */

/**
 * Check if the error is below the convergence tolerance.
 *
 * @param error      Current error value
 * @param tolerance  Convergence threshold
 * @return           1 if converged (error < tolerance), 0 otherwise
 */
int check_convergence(double error, double tolerance);

/**
 * Check if convergence was achieved within iteration limit.
 *
 * @param result  JacobiResult from solver
 * @return        1 if converged, 0 otherwise
 */
int is_converged(JacobiResult result);

/* --------------------------------------------------------------------------
 * Validation Against Baseline
 * -------------------------------------------------------------------------- */

/**
 * Validate a test grid against a baseline (serial reference) grid.
 *
 * Computes multiple error metrics and determines if the grids match
 * within the specified tolerance.
 *
 * @param test_grid      Grid from parallel implementation
 * @param baseline_grid  Grid from serial reference solver
 * @param N              Grid dimension
 * @param tolerance      Acceptable error threshold
 * @return               ValidationResult with all metrics
 */
ValidationResult validate_against_baseline(double **test_grid,
                                           double **baseline_grid,
                                           int N,
                                           double tolerance);

/* --------------------------------------------------------------------------
 * Reporting and Output
 * -------------------------------------------------------------------------- */

/**
 * Print validation results in a formatted table.
 *
 * @param result  ValidationResult to display
 */
void print_validation_result(ValidationResult result);

/**
 * Print convergence information from solver result.
 *
 * @param result     JacobiResult from solver
 * @param N          Grid dimension
 * @param tolerance  Convergence tolerance used
 */
void print_convergence_info(JacobiResult result, int N, double tolerance);

/**
 * Print a summary comparing two solver results (e.g., serial vs parallel).
 *
 * @param serial_result    Result from serial solver
 * @param parallel_result  Result from parallel solver
 * @param label            Name of parallel implementation (e.g., "OpenMP", "MPI")
 */
void print_comparison_summary(JacobiResult serial_result,
                              JacobiResult parallel_result,
                              const char *label);

/**
 * Check if boundaries match between two grids.
 * Useful for verifying that parallel implementations preserve boundary conditions.
 *
 * @param grid1  First grid (N×N)
 * @param grid2  Second grid (N×N)
 * @param N      Grid dimension
 * @return       1 if all boundaries match exactly, 0 otherwise
 */
int validate_boundaries(double **grid1, double **grid2, int N);

#endif /* CONVERGENCE_H */
