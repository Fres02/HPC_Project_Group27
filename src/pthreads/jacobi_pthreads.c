#define _POSIX_C_SOURCE 200112L  /* For pthread_barrier */

#include "jacobi_pthreads.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

/* Thread synchronization primitives */
static pthread_barrier_t barrier;
static double global_max_error = 0.0;

/* Shared data passed to thread worker function */
typedef struct {
    double **curr;        /* current grid (read) */
    double **next;        /* next grid (write) */
    int N;                /* grid dimension */
    int num_threads;      /* total number of threads */
    int thread_id;        /* this thread's ID (0 to num_threads-1) */
    double local_error;   /* thread-local error computation */
} ThreadArgs;

/* --------------------------------------------------------------------------
 * Worker function for each thread
 * 
 * Each thread is assigned a contiguous range of rows to update.
 * Threads synchronize at barriers between iterations.
 * -------------------------------------------------------------------------- */
static void *jacobi_worker(void *arg)
{
    ThreadArgs *args = (ThreadArgs *)arg;

    double **curr = args->curr;
    double **next = args->next;
    int N = args->N;
    int num_threads = args->num_threads;
    int thread_id = args->thread_id;

    /* Calculate row range for this thread */
    int rows_per_thread = (N - 2) / num_threads;  /* exclude top and bottom boundaries */
    int extra_rows = (N - 2) % num_threads;

    int start_row = 1 + thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);

    /* Update assigned rows */
    double thread_max_error = 0.0;
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < N - 1; j++) {
            double new_val = 0.25 * (curr[i-1][j] +
                                     curr[i+1][j] +
                                     curr[i][j-1] +
                                     curr[i][j+1]);
            next[i][j] = new_val;

            double diff = fabs(new_val - curr[i][j]);
            if (diff > thread_max_error)
                thread_max_error = diff;
        }
    }

    args->local_error = thread_max_error;

    pthread_exit(NULL);
}

/* --------------------------------------------------------------------------
 * Main POSIX Threads Jacobi solver
 * -------------------------------------------------------------------------- */
PthreadsJacobiResult jacobi_solve_pthreads(double **host_grid, int N,
                                           int num_threads,
                                           int max_iter, double tol)
{
    PthreadsJacobiResult result = {0, 0.0, 0};

    if (num_threads <= 0 || num_threads > N - 2) {
        fprintf(stderr, "jacobi_solve_pthreads: invalid num_threads=%d for N=%d\n",
                num_threads, N);
        return result;
    }

    /* Allocate second buffer for double-buffering */
    double **buf_b = alloc_grid(N);
    if (!buf_b) {
        fprintf(stderr, "jacobi_solve_pthreads: failed to allocate buffer\n");
        return result;
    }
    copy_grid(buf_b, host_grid, N);

    double **curr = host_grid;
    double **next = buf_b;

    /* Create thread array and argument structures */
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs *thread_args = (ThreadArgs *)malloc(num_threads * sizeof(ThreadArgs));

    if (!threads || !thread_args) {
        fprintf(stderr, "jacobi_solve_pthreads: failed to allocate thread structures\n");
        free(threads);
        free(thread_args);
        free_grid(buf_b, N);
        return result;
    }

    /* Initialize barrier for thread synchronization */
    if (pthread_barrier_init(&barrier, NULL, num_threads) != 0) {
        fprintf(stderr, "jacobi_solve_pthreads: failed to initialize barrier\n");
        free(threads);
        free(thread_args);
        free_grid(buf_b, N);
        return result;
    }

    fprintf(stderr, "jacobi_solve_pthreads: starting with N=%d, threads=%d, max_iter=%d, tol=%g\n",
            N, num_threads, max_iter, tol);

    for (int iter = 0; iter < max_iter; iter++) {

        global_max_error = 0.0;

        /* Initialize thread arguments */
        for (int t = 0; t < num_threads; t++) {
            thread_args[t].curr = curr;
            thread_args[t].next = next;
            thread_args[t].N = N;
            thread_args[t].num_threads = num_threads;
            thread_args[t].thread_id = t;
            thread_args[t].local_error = 0.0;
        }

        /* Launch threads */
        for (int t = 0; t < num_threads; t++) {
            if (pthread_create(&threads[t], NULL, jacobi_worker, &thread_args[t]) != 0) {
                fprintf(stderr, "jacobi_solve_pthreads: failed to create thread %d\n", t);
                result.iterations = iter;
                result.final_error = global_max_error;
                goto cleanup;
            }
        }

        /* Wait for all threads to complete */
        for (int t = 0; t < num_threads; t++) {
            if (pthread_join(threads[t], NULL) != 0) {
                fprintf(stderr, "jacobi_solve_pthreads: failed to join thread %d\n", t);
            }
        }

        /* Compute global maximum error from thread-local errors */
        double max_err = 0.0;
        for (int t = 0; t < num_threads; t++) {
            if (thread_args[t].local_error > max_err)
                max_err = thread_args[t].local_error;
        }

        /* Swap buffers */
        double **temp = curr;
        curr = next;
        next = temp;

        result.iterations = iter + 1;
        result.final_error = max_err;

        if (iter % 100 == 0 || iter < 5) {
            fprintf(stderr, "  iter %d: max_err = %g\n", iter, max_err);
        }

        if (max_err < tol) {
            result.converged = 1;
            fprintf(stderr, "jacobi_solve_pthreads: converged in %d iterations (final error = %g)\n",
                    result.iterations, result.final_error);
            break;
        }
    }

    if (!result.converged) {
        fprintf(stderr, "jacobi_solve_pthreads: reached max_iter=%d without convergence (final error = %g)\n",
                max_iter, result.final_error);
    }

    /* Ensure result lives in host_grid (caller's buffer) */
    if (curr != host_grid)
        copy_grid(host_grid, curr, N);

cleanup:
    /* Clean up resources */
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(thread_args);
    free_grid(buf_b, N);

    return result;
}
