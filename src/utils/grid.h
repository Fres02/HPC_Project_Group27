#ifndef GRID_H
#define GRID_H

/* Default boundary condition values (fixed Dirichlet) */
#define BC_TOP    1.0
#define BC_BOTTOM 0.0
#define BC_LEFT   0.0
#define BC_RIGHT  0.0

/*
 * Allocate and return a flat N×N grid (row-major, contiguous storage).
 * Returns NULL on allocation failure.
 * Caller must free with free_grid().
 */
double **alloc_grid(int N);

/*
 * Free a grid allocated with alloc_grid().
 * N is accepted for API symmetry but is not used.
 */
void free_grid(double **grid, int N);

/*
 * Initialise the grid: fill interior with 0.0 and apply the default
 * Dirichlet boundary conditions (BC_TOP, BC_BOTTOM, BC_LEFT, BC_RIGHT).
 * Corners belong to the top/bottom rows.
 */
void init_grid(double **grid, int N);

/*
 * Like init_grid but with caller-supplied boundary values and interior fill.
 * Interior points are set to interior_val; corners belong to top/bottom rows.
 */
void init_grid_custom(double **grid, int N,
                      double top, double bottom,
                      double left, double right,
                      double interior_val);

/*
 * Set every interior point (rows 1..N-2, cols 1..N-2) to value.
 * Boundary values are left unchanged.
 */
void fill_interior(double **grid, int N, double value);

/*
 * Swap two grid pointer-arrays in O(1).
 * After the call *a points to what *b pointed to, and vice-versa.
 */
void swap_buffers(double ***a, double ***b);

/*
 * Copy the full N×N data from src into dst (both allocated with alloc_grid(N)).
 */
void copy_grid(double **dst, double **src, int N);

#endif /* GRID_H */
