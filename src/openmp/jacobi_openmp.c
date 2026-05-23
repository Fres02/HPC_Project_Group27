#include "jacobi_openmp.h"

#include <math.h>
#include <stdio.h>
#include <omp.h>

JacobiResult jacobi_solve_openmp(double **grid, int N, int max_iter, double tol, int num_threads)
{
    JacobiResult result = {0, 0.0, 0};
    
    /* Set number of threads if specified */
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    /* Allocate second buffer for double-buffering */
    double **buf_b = alloc_grid(N);
    if (!buf_b) {
        fprintf(stderr, "jacobi_solve_openmp: failed to allocate buffer\n");
        return result;
    }
    copy_grid(buf_b, grid, N);
    
    /* curr - buffer we READ from each sweep
       next - buffer we WRITE into each sweep */
    double **curr = grid;
    double **next = buf_b;
    
    for (int iter = 0; iter < max_iter; iter++) {
        
        double max_err = 0.0;
        
        /* Parallelize the stencil computation with OpenMP.
         * Use reduction to find maximum error across all threads.
         * collapse(2) combines the two nested loops for better load balancing. */
        #pragma omp parallel for collapse(2) reduction(max:max_err) schedule(static)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                /* 5-point stencil: average of 4 neighbors */
                next[i][j] = 0.25 * (curr[i-1][j] +
                                     curr[i+1][j] +
                                     curr[i][j-1] +
                                     curr[i][j+1]);
                
                /* Compute local error */
                double diff = fabs(next[i][j] - curr[i][j]);
                
                /* Reduction finds the maximum across all threads */
                if (diff > max_err) {
                    max_err = diff;
                }
            }
        }
        
        /* Swap buffers (pointer swap is O(1) and thread-safe) */
        swap_buffers(&curr, &next);
        
        result.iterations  = iter + 1;
        result.final_error = max_err;
        
        /* Check convergence */
        if (max_err < tol) {
            result.converged = 1;
            break;
        }
    }
    
    /* Ensure result is in the original grid buffer */
    if (curr != grid) {
        copy_grid(grid, curr, N);
    }
    
    free_grid(buf_b, N);
    return result;
}
