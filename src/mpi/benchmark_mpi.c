/*
 * MPI Jacobi Solver - Benchmark and Validation Program
 *
 * Steps:
 *   1. Run the serial baseline (rank 0)
 *   2. Run the MPI solver under the launched rank count
 *   3. Validate correctness vs serial baseline (rank 0)
 *   4. Report timing, speedup, efficiency, RMSE
 *
 * Run with:
 *   mpiexec -n <P> ./benchmark_mpi.exe [N] [max_iter] [tol]
 *
 * For thread/process scaling, run multiple times with different -n.
 */

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

#include "../core/grid.h"
#include "../core/jacobi_serial.h"
#include "../core/convergence.h"
#include "jacobi_mpi.h"

static void print_header(const char *title) {
    printf("\n");
    printf("==================================================\n");
    printf("  %s\n", title);
    printf("==================================================\n");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    N        = (argc > 1) ? atoi(argv[1]) : 200;
    int    max_iter = (argc > 2) ? atoi(argv[2]) : 10000;
    double tol      = (argc > 3) ? atof(argv[3]) : 1e-6;

    if (rank == 0) {
        print_header("MPI Jacobi Solver - Benchmark & Validation");
        printf("Configuration:\n");
        printf("  Grid Size       : %d x %d\n", N, N);
        printf("  Interior Points : %d\n", (N-2) * (N-2));
        printf("  Max Iterations  : %d\n", max_iter);
        printf("  Tolerance       : %.2e\n", tol);
        printf("  MPI Ranks       : %d\n", size);
        printf("\n");
    }

    /* -------- Serial baseline -------- */
    double **serial_grid = NULL;
    JacobiResult serial_result = {0, 0.0, 0};
    double t_serial = 0.0;

    if (rank == 0) {
        print_header("Step 1: Serial Baseline Solver");
        serial_grid = alloc_grid(N);
        if (!serial_grid) {
            fprintf(stderr, "Error: Failed to allocate serial grid\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        init_grid(serial_grid, N);

        double t0 = MPI_Wtime();
        serial_result = jacobi_solve(serial_grid, N, max_iter, tol);
        t_serial = MPI_Wtime() - t0;

        print_convergence_info(serial_result, N, tol);
        printf("  Execution Time  : %.6f seconds\n", t_serial);
    }

    /* -------- MPI version -------- */
    double **mpi_grid = NULL;
    if (rank == 0) {
        print_header("Step 2: MPI Parallel Solver");
        mpi_grid = alloc_grid(N);
        if (!mpi_grid) {
            fprintf(stderr, "Error: Failed to allocate MPI grid\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        init_grid(mpi_grid, N);
        printf("Running MPI solver with %d ranks...\n", size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    JacobiResult mpi_result = jacobi_solve_mpi(mpi_grid, N, max_iter, tol, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_mpi = MPI_Wtime() - t0;

    if (rank == 0) {
        print_convergence_info(mpi_result, N, tol);
        printf("  Execution Time  : %.6f seconds\n", t_mpi);

        /* -------- Validation -------- */
        print_header("Step 3: Validation");
        ValidationResult val = validate_against_baseline(mpi_grid, serial_grid, N, tol);
        print_validation_result(val);
        int boundaries_ok = validate_boundaries(mpi_grid, serial_grid, N);
        printf("Boundary Validation: %s\n", boundaries_ok ? "PASS" : "FAIL");

        /* -------- Comparison -------- */
        print_comparison_summary(serial_result, mpi_result, "MPI");

        /* -------- Performance -------- */
        print_header("Step 4: Performance Analysis");
        double speedup    = (t_mpi > 0) ? (t_serial / t_mpi) : 0.0;
        double efficiency = (size > 0) ? (speedup / size) * 100.0 : 0.0;
        printf("Timing Results:\n");
        printf("  Serial Time   : %.6f s\n", t_serial);
        printf("  MPI Time      : %.6f s (with %d ranks)\n", t_mpi, size);
        printf("  Speedup       : %.2fx\n", speedup);
        printf("  Efficiency    : %.2f%%\n", efficiency);
        printf("\n");

        /* -------- Report data summary -------- */
        print_header("Step 5: Data for Analysis Report");
        printf("Accuracy Metrics:\n");
        printf("  RMSE                 : %.10e\n", val.rmse);
        printf("  Maximum Error        : %.10e\n", val.max_error);
        printf("  Mean Absolute Error  : %.10e\n", val.mae);
        printf("  Validation Status    : %s\n", val.grids_match ? "PASS" : "FAIL");
        printf("\n");
        printf("Performance Metrics:\n");
        printf("  Number of Ranks      : %d\n", size);
        printf("  Serial Time          : %.6f s\n", t_serial);
        printf("  Parallel Time        : %.6f s\n", t_mpi);
        printf("  Speedup              : %.2fx\n", speedup);
        printf("  Parallel Efficiency  : %.2f%%\n", efficiency);
        printf("\n");

        print_header("Final Status");
        if (val.grids_match && boundaries_ok) {
            printf("  [PASS] All validations PASSED\n");
            printf("  [PASS] MPI implementation matches serial baseline (RMSE within tol)\n");
            printf("  [PASS] Speedup achieved: %.2fx with %d ranks\n", speedup, size);
        } else {
            printf("  [FAIL] Validation FAILED\n");
            if (!val.grids_match)   printf("  [FAIL] Grids do not match within tolerance\n");
            if (!boundaries_ok)     printf("  [FAIL] Boundaries not preserved correctly\n");
        }
        printf("==================================================\n\n");

        free_grid(serial_grid, N);
        free_grid(mpi_grid, N);

        MPI_Finalize();
        return val.grids_match ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
