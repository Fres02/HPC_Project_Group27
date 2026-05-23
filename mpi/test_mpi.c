/*
 * Simple test program for the MPI Jacobi solver.
 * Quick validation against the serial baseline.
 *
 * Run with:
 *   mpiexec -n <P> ./test_mpi.exe [N] [max_iter] [tol]
 */

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

#include "../core/grid.h"
#include "../core/jacobi_serial.h"
#include "../core/convergence.h"
#include "jacobi_mpi.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    N        = (argc > 1) ? atoi(argv[1]) : 100;
    int    max_iter = (argc > 2) ? atoi(argv[2]) : 10000;
    double tol      = (argc > 3) ? atof(argv[3]) : 1e-6;

    if (rank == 0) {
        printf("==========================================\n");
        printf("  MPI Jacobi Solver - Quick Test\n");
        printf("==========================================\n");
        printf("  Grid Size  : %d x %d\n", N, N);
        printf("  Tolerance  : %.2e\n", tol);
        printf("  Max Iters  : %d\n", max_iter);
        printf("  MPI Ranks  : %d\n", size);
        printf("==========================================\n\n");
    }

    /* -------- Serial baseline on rank 0 -------- */
    double **serial_grid = NULL;
    JacobiResult serial_result = {0, 0.0, 0};
    double t_serial = 0.0;

    if (rank == 0) {
        printf("Running serial solver on rank 0...\n");
        serial_grid = alloc_grid(N);
        if (!serial_grid) {
            fprintf(stderr, "Failed to allocate serial grid\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        init_grid(serial_grid, N);

        double t0 = MPI_Wtime();
        serial_result = jacobi_solve(serial_grid, N, max_iter, tol);
        t_serial = MPI_Wtime() - t0;

        printf("  Iterations : %d\n", serial_result.iterations);
        printf("  Final Error: %.10e\n", serial_result.final_error);
        printf("  Converged  : %s\n", serial_result.converged ? "YES" : "NO");
        printf("  Time       : %.6f s\n\n", t_serial);
    }

    /* -------- MPI version -------- */
    double **mpi_grid = NULL;
    if (rank == 0) {
        mpi_grid = alloc_grid(N);
        if (!mpi_grid) {
            fprintf(stderr, "Failed to allocate MPI grid\n");
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
        printf("  Iterations : %d\n", mpi_result.iterations);
        printf("  Final Error: %.10e\n", mpi_result.final_error);
        printf("  Converged  : %s\n", mpi_result.converged ? "YES" : "NO");
        printf("  Time       : %.6f s\n\n", t_mpi);

        printf("Validating...\n");
        ValidationResult val = validate_against_baseline(mpi_grid, serial_grid, N, tol);
        printf("  RMSE      : %.10e\n", val.rmse);
        printf("  Max Error : %.10e\n", val.max_error);
        printf("  Status    : %s\n\n", val.grids_match ? "PASS" : "FAIL");

        if (t_mpi > 0) {
            double speedup    = t_serial / t_mpi;
            double efficiency = (speedup / size) * 100.0;
            printf("Performance:\n");
            printf("  Speedup    : %.2fx\n", speedup);
            printf("  Efficiency : %.2f%%\n\n", efficiency);
        }

        if (val.grids_match) {
            printf("[PASS] MPI implementation matches serial baseline.\n");
        } else {
            printf("[FAIL] Results do not match serial baseline.\n");
        }
        printf("==========================================\n");

        free_grid(serial_grid, N);
        free_grid(mpi_grid, N);

        MPI_Finalize();
        return val.grids_match ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
