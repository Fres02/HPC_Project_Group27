#include "jacobi_cuda.h"

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

/* Error handling macro for CUDA calls */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at line %d\n", \
                    cudaGetErrorString(err), __LINE__); \
            return (CudaJacobiResult){0, 0.0, 0}; \
        } \
    } while(0)

/* --------------------------------------------------------------------------
 * CUDA Kernel: Jacobi update for interior points
 * 
 * Each thread updates one grid point using the average of its 4 neighbors.
 * Reads from 'curr' and writes to 'next'.
 * Also computes the absolute difference for convergence checking.
 * -------------------------------------------------------------------------- */
__global__ void jacobi_update_kernel(double *curr, double *next,
                                      double *d_error_workspace,
                                      int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  /* row, skip boundary */
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;  /* col, skip boundary */

    if (i < N - 1 && j < N - 1) {
        int idx    = i * N + j;
        int idx_up = (i - 1) * N + j;
        int idx_dn = (i + 1) * N + j;
        int idx_lf = i * N + (j - 1);
        int idx_rt = i * N + (j + 1);

        double new_val = 0.25 * (curr[idx_up] +
                                 curr[idx_dn] +
                                 curr[idx_lf] +
                                 curr[idx_rt]);
        next[idx] = new_val;

        double diff = fabs(new_val - curr[idx]);
        d_error_workspace[idx] = diff;
    }
}

/* --------------------------------------------------------------------------
 * CUDA Kernel: Parallel reduction to find maximum error
 * 
 * Reduces the error workspace array to compute the global maximum.
 * Uses shared memory for efficient block-level reduction.
 * -------------------------------------------------------------------------- */
__global__ void reduce_max_error_kernel(double *d_error_workspace,
                                        double *d_max_error,
                                        int N)
{
    extern __shared__ double sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    /* Load data into shared memory */
    if (idx < N * N) {
        sdata[tid] = d_error_workspace[idx];
    } else {
        sdata[tid] = 0.0;
    }
    __syncthreads();

    /* Parallel reduction within block */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    /* Write block result to global memory */
    if (tid == 0) {
        d_max_error[blockIdx.x] = sdata[0];
    }
}

/* --------------------------------------------------------------------------
 * Copy host grid to device (row-major, contiguous)
 * -------------------------------------------------------------------------- */
static void copy_host_to_device(double **h_grid, double *d_grid, int N)
{
    /* Allocate temporary CPU buffer to linearize the grid */
    double *h_linear = (double *)malloc(N * N * sizeof(double));
    if (!h_linear) {
        fprintf(stderr, "copy_host_to_device: malloc failed\n");
        return;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_linear[i * N + j] = h_grid[i][j];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_grid, h_linear, N * N * sizeof(double),
                          cudaMemcpyHostToDevice));

    free(h_linear);
}

/* --------------------------------------------------------------------------
 * Copy device grid back to host
 * -------------------------------------------------------------------------- */
static void copy_device_to_host(double *d_grid, double **h_grid, int N)
{
    double *h_linear = (double *)malloc(N * N * sizeof(double));
    if (!h_linear) {
        fprintf(stderr, "copy_device_to_host: malloc failed\n");
        return;
    }

    CUDA_CHECK(cudaMemcpy(h_linear, d_grid, N * N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_grid[i][j] = h_linear[i * N + j];
        }
    }

    free(h_linear);
}

/* --------------------------------------------------------------------------
 * Main CUDA Jacobi solver
 * -------------------------------------------------------------------------- */
CudaJacobiResult jacobi_solve_cuda(double **host_grid, int N, int max_iter, double tol)
{
    CudaJacobiResult result = {0, 0.0, 0};

    /* Device pointers */
    double *d_curr = NULL, *d_next = NULL, *d_error_workspace = NULL, *d_max_error = NULL;
    double *d_temp = NULL;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc(&d_curr, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_next, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_error_workspace, N * N * sizeof(double)));

    /* Allocate workspace for reduction (one value per block) */
    int max_blocks = (N * N + 255) / 256;
    CUDA_CHECK(cudaMalloc(&d_max_error, max_blocks * sizeof(double)));

    /* Copy initial grid to device */
    copy_host_to_device(host_grid, d_curr, N);
    CUDA_CHECK(cudaMemcpy(d_next, d_curr, N * N * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    /* Configure grid and block dimensions for stencil kernel */
    dim3 block_dim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 grid_dim((N + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE,
                  (N + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

    fprintf(stderr, "jacobi_solve_cuda: starting with N=%d, max_iter=%d, tol=%g\n",
            N, max_iter, tol);

    for (int iter = 0; iter < max_iter; iter++) {

        /* Launch stencil update kernel */
        jacobi_update_kernel<<<grid_dim, block_dim>>>(d_curr, d_next, d_error_workspace, N);
        CUDA_CHECK(cudaGetLastError());

        /* Launch reduction kernel to find maximum error */
        int reduction_blocks = (N * N + 255) / 256;
        int reduction_threads = 256;
        reduce_max_error_kernel<<<reduction_blocks, reduction_threads,
                                 reduction_threads * sizeof(double)>>>(d_error_workspace,
                                                                       d_max_error,
                                                                       N);
        CUDA_CHECK(cudaGetLastError());

        /* Additional reduction on GPU to find global max */
        double h_block_errors[max_blocks];
        CUDA_CHECK(cudaMemcpy(h_block_errors, d_max_error,
                              reduction_blocks * sizeof(double),
                              cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (int b = 0; b < reduction_blocks; b++) {
            if (h_block_errors[b] > max_err)
                max_err = h_block_errors[b];
        }

        /* Swap buffers on device */
        d_temp = d_curr;
        d_curr = d_next;
        d_next = d_temp;

        result.iterations  = iter + 1;
        result.final_error = max_err;

        if (iter % 100 == 0 || iter < 5) {
            fprintf(stderr, "  iter %d: max_err = %g\n", iter, max_err);
        }

        if (max_err < tol) {
            result.converged = 1;
            fprintf(stderr, "jacobi_solve_cuda: converged in %d iterations (final error = %g)\n",
                    result.iterations, result.final_error);
            break;
        }
    }

    if (!result.converged) {
        fprintf(stderr, "jacobi_solve_cuda: reached max_iter=%d without convergence (final error = %g)\n",
                max_iter, result.final_error);
    }

    /* Copy result back to host */
    copy_device_to_host(d_curr, host_grid, N);

    /* Free device memory */
    cudaFree(d_curr);
    cudaFree(d_next);
    cudaFree(d_error_workspace);
    cudaFree(d_max_error);

    return result;
}
