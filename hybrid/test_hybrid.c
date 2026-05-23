/*
 * Simple test program for the Hybrid MPI+OpenMP Jacobi solver.
 * Quick validation against the serial baseline.
 *
 * Run with:
 *   mpiexec -n <P> ./test_hybrid.exe [N] [max_iter] [tol] [threads_per_rank]
 */

#include <mpi.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>

#include "../core/grid.h"
#include "../core/jacobi_serial.h"
#include "../core/convergence.h"
#include "jacobi_hybrid.h"

int main(int argc, char **argv)
{
    /* Initialize MPI with thread support (FUNNELED is enough: only the main
     * thread of each rank makes MPI calls; OpenMP threads only run inside
     * the inner sweep). */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    N           = (argc > 1) ? atoi(argv[1]) : 100;
    int    max_iter    = (argc > 2) ? atoi(argv[2]) : 10000;
    double tol         = (argc > 3) ? atof(argv[3]) : 1e-6;
    int    num_threads = (argc > 4) ? atoi(argv[4]) : 0; /* 0 = default */

    int threads_used = (num_threads > 0) ? num_threads : omp_get_max_threads();

    if (rank == 0) {
        printf("==========================================\n");
        printf("  Hybrid MPI+OpenMP Jacobi - Quick Test\n");
        printf("==========================================\n");
        printf("  Grid Size       : %d x %d\n", N, N);
        printf("  Tolerance       : %.2e\n", tol);
        printf("  Max Iters       : %d\n", max_iter);
        printf("  MPI Ranks       : %d\n", size);
        printf("  Threads/Rank    : %d\n", threads_used);
        printf("  Total Workers   : %d\n", size * threads_used);
        printf("  MPI Thread Lvl  : %s\n",
               provided == MPI_THREAD_FUNNELED  ? "FUNNELED"  :
               provided == MPI_THREAD_SERIALIZED? "SERIALIZED":
               provided == MPI_THREAD_MULTIPLE  ? "MULTIPLE"  : "SINGLE");
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

    /* -------- Hybrid version -------- */
    double **hybrid_grid = NULL;
    if (rank == 0) {
        hybrid_grid = alloc_grid(N);
        if (!hybrid_grid) {
            fprintf(stderr, "Failed to allocate hybrid grid\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        init_grid(hybrid_grid, N);
        printf("Running hybrid solver (%d ranks x %d threads = %d workers)...\n",
               size, threads_used, size * threads_used);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    JacobiResult hyb_result = jacobi_solve_hybrid(hybrid_grid, N, max_iter, tol,
                                                  num_threads, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_hyb = MPI_Wtime() - t0;

    if (rank == 0) {
        printf("  Iterations : %d\n", hyb_result.iterations);
        printf("  Final Error: %.10e\n", hyb_result.final_error);
        printf("  Converged  : %s\n", hyb_result.converged ? "YES" : "NO");
        printf("  Time       : %.6f s\n\n", t_hyb);

        printf("Validating...\n");
        ValidationResult val = validate_against_baseline(hybrid_grid, serial_grid, N, tol);
        printf("  RMSE      : %.10e\n", val.rmse);
        printf("  Max Error : %.10e\n", val.max_error);
        printf("  Status    : %s\n\n", val.grids_match ? "PASS" : "FAIL");

        if (t_hyb > 0) {
            int total_workers = size * threads_used;
            double speedup    = t_serial / t_hyb;
            double efficiency = (speedup / total_workers) * 100.0;
            printf("Performance:\n");
            printf("  Speedup    : %.2fx\n", speedup);
            printf("  Efficiency : %.2f%% (vs %d total workers)\n\n", efficiency, total_workers);
        }

        if (val.grids_match) {
            printf("[PASS] Hybrid implementation matches serial baseline.\n");
        } else {
            printf("[FAIL] Results do not match serial baseline.\n");
        }
        printf("==========================================\n");

        free_grid(serial_grid, N);
        free_grid(hybrid_grid, N);

        MPI_Finalize();
        return val.grids_match ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
