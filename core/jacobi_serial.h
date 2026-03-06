#ifndef JACOBI_SERIAL_H
#define JACOBI_SERIAL_H

/* Default solver parameters */
#define DEFAULT_N       100
#define DEFAULT_MAX_ITER 10000
#define DEFAULT_TOL     1e-6

/* Boundary values (fixed Dirichlet conditions) */
#define BC_TOP    1.0
#define BC_BOTTOM 0.0
#define BC_LEFT   0.0
#define BC_RIGHT  0.0

/* Result struct returned by the solver */
typedef struct {
    int    iterations;   /* number of iterations performed */
    double final_error;  /* residual error at termination  */
    int    converged;    /* 1 if tolerance was met, 0 if max_iter hit */
} JacobiResult;

/*
 * Allocate and return a flat N×N grid (row-major).
 * Caller must free with free_grid().
 */
double **alloc_grid(int N);
void     free_grid(double **grid, int N);

/*
 * Fill the interior with 0.0 and impose fixed Dirichlet
 * boundary conditions on the four edges.
 */
void init_grid(double **grid, int N);

/*
 * Run the Jacobi iteration.
 *
 *   grid    – initialised N×N grid (boundary already set)
 *   N       – grid dimension
 *   max_iter – iteration cap
 *   tol     – convergence tolerance (max absolute change per sweep)
 *
 * Returns a JacobiResult with iteration count, final error, and
 * a convergence flag.
 */
JacobiResult jacobi_solve(double **grid, int N, int max_iter, double tol);

#endif /* JACOBI_SERIAL_H */
