/*
 * Hybrid MPI+OpenMP Jacobi Solver - Benchmark and Validation Program
 *
 * Steps:
 *   1. Run the serial baseline (rank 0)
 *   2. Run the hybrid solver under the launched rank count x threads/rank
 *   3. Validate correctness vs serial baseline (rank 0)
 *   4. Report timing, speedup, efficiency, RMSE
 *
 * Run with:
 *   mpiexec -n <P> ./benchmark_hybrid.exe [N] [max_iter] [tol] [threads_per_rank]
 *
 * For scaling studies, vary P (ranks) and threads_per_rank independently.
 */

#include <mpi.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>

#include "../core/grid.h"
#include "../core/jacobi_serial.h"
#include "../core/convergence.h"
#include "jacobi_hybrid.h"

static void print_header(const char *title) {
    printf("\n");
    printf("==================================================\n");
    printf("  %s\n", title);
    printf("==================================================\n");
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    N           = (argc > 1) ? atoi(argv[1]) : 200;
    int    max_iter    = (argc > 2) ? atoi(argv[2]) : 10000;
    double tol         = (argc > 3) ? atof(argv[3]) : 1e-6;
    int    num_threads = (argc > 4) ? atoi(argv[4]) : 0;

    int threads_used  = (num_threads > 0) ? num_threads : omp_get_max_threads();
    int total_workers = size * threads_used;

    if (rank == 0) {
        print_header("Hybrid MPI+OpenMP Jacobi - Benchmark & Validation");
        printf("Configuration:\n");
        printf("  Grid Size       : %d x %d\n", N, N);
        printf("  Interior Points : %d\n", (N-2) * (N-2));
        printf("  Max Iterations  : %d\n", max_iter);
        printf("  Tolerance       : %.2e\n", tol);
        printf("  MPI Ranks       : %d\n", size);
        printf("  Threads/Rank    : %d\n", threads_used);
        printf("  Total Workers   : %d\n", total_workers);
        printf("  MPI Thread Lvl  : %s\n",
               provided == MPI_THREAD_FUNNELED  ? "FUNNELED"  :
               provided == MPI_THREAD_SERIALIZED? "SERIALIZED":
               provided == MPI_THREAD_MULTIPLE  ? "MULTIPLE"  : "SINGLE");
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

    /* -------- Hybrid version -------- */
    double **hybrid_grid = NULL;
    if (rank == 0) {
        print_header("Step 2: Hybrid MPI+OpenMP Parallel Solver");
        hybrid_grid = alloc_grid(N);
        if (!hybrid_grid) {
            fprintf(stderr, "Error: Failed to allocate hybrid grid\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        init_grid(hybrid_grid, N);
        printf("Running hybrid solver (%d ranks x %d threads = %d workers)...\n",
               size, threads_used, total_workers);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    JacobiResult hyb_result = jacobi_solve_hybrid(hybrid_grid, N, max_iter, tol,
                                                  num_threads, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_hyb = MPI_Wtime() - t0;

    if (rank == 0) {
        print_convergence_info(hyb_result, N, tol);
        printf("  Execution Time  : %.6f seconds\n", t_hyb);

        /* -------- Validation -------- */
        print_header("Step 3: Validation");
        ValidationResult val = validate_against_baseline(hybrid_grid, serial_grid, N, tol);
        print_validation_result(val);
        int boundaries_ok = validate_boundaries(hybrid_grid, serial_grid, N);
        printf("Boundary Validation: %s\n", boundaries_ok ? "PASS" : "FAIL");

        /* -------- Comparison -------- */
        print_comparison_summary(serial_result, hyb_result, "Hybrid");

        /* -------- Performance -------- */
        print_header("Step 4: Performance Analysis");
        double speedup    = (t_hyb > 0)        ? (t_serial / t_hyb)              : 0.0;
        double efficiency = (total_workers > 0) ? (speedup / total_workers) * 100 : 0.0;
        printf("Timing Results:\n");
        printf("  Serial Time   : %.6f s\n", t_serial);
        printf("  Hybrid Time   : %.6f s (%d ranks x %d threads)\n", t_hyb, size, threads_used);
        printf("  Speedup       : %.2fx\n", speedup);
        printf("  Efficiency    : %.2f%% (vs %d total workers)\n", efficiency, total_workers);
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
        printf("  Ranks                : %d\n", size);
        printf("  Threads / Rank       : %d\n", threads_used);
        printf("  Total Workers        : %d\n", total_workers);
        printf("  Serial Time          : %.6f s\n", t_serial);
        printf("  Parallel Time        : %.6f s\n", t_hyb);
        printf("  Speedup              : %.2fx\n", speedup);
        printf("  Parallel Efficiency  : %.2f%%\n", efficiency);
        printf("\n");

        print_header("Final Status");
        if (val.grids_match && boundaries_ok) {
            printf("  [PASS] All validations PASSED\n");
            printf("  [PASS] Hybrid implementation matches serial baseline (RMSE within tol)\n");
            printf("  [PASS] Speedup achieved: %.2fx with %d ranks x %d threads = %d workers\n",
                   speedup, size, threads_used, total_workers);
        } else {
            printf("  [FAIL] Validation FAILED\n");
            if (!val.grids_match)   printf("  [FAIL] Grids do not match within tolerance\n");
            if (!boundaries_ok)     printf("  [FAIL] Boundaries not preserved correctly\n");
        }
        printf("==================================================\n\n");

        free_grid(serial_grid, N);
        free_grid(hybrid_grid, N);

        MPI_Finalize();
        return val.grids_match ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
