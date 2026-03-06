/*
 * test_jacobi_serial.c
 *
 * Unit tests for the Feature-1 serial Jacobi solver.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -o test_jacobi_serial \
 *       test_jacobi_serial.c jacobi_serial.c -lm
 *
 * Run:
 *   ./test_jacobi_serial
 */

#include "jacobi_serial.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Minimal test harness
 * ------------------------------------------------------------------------- */

static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        tests_run++;                                                          \
        if (cond) {                                                           \
            tests_passed++;                                                   \
            printf("  [PASS] %s\n", msg);                                    \
        } else {                                                              \
            printf("  [FAIL] %s\n", msg);                                    \
        }                                                                     \
    } while (0)

/* -------------------------------------------------------------------------
 * Test 1 – alloc_grid / free_grid
 *   Verify that the allocator returns a non-NULL, correctly sized grid and
 *   that free_grid does not crash.
 * ------------------------------------------------------------------------- */
static void test_alloc_free(void)
{
    printf("\n[Test 1] Grid allocation and deallocation\n");

    int N = 10;
    double **g = alloc_grid(N);
    CHECK(g != NULL, "alloc_grid returns non-NULL for N=10");

    /* Row pointers must be N apart (contiguous layout) */
    int contiguous = 1;
    for (int i = 1; i < N; i++)
        if (g[i] != g[0] + (size_t)i * N) { contiguous = 0; break; }
    CHECK(contiguous, "Row pointers are contiguous (row-major layout)");

    free_grid(g, N); /* must not crash */
    CHECK(1, "free_grid completes without crash");
}

/* -------------------------------------------------------------------------
 * Test 2 – init_grid boundary conditions
 *   After init_grid the four edges must match the BC constants and the
 *   interior must be 0.
 * ------------------------------------------------------------------------- */
static void test_boundary_conditions(void)
{
    printf("\n[Test 2] Boundary conditions after init_grid\n");

    int N = 8;
    double **g = alloc_grid(N);
    init_grid(g, N);

    /* Top row == BC_TOP */
    int top_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[0][j] - BC_TOP) > 1e-15) { top_ok = 0; break; }
    CHECK(top_ok, "Top boundary == BC_TOP");

    /* Bottom row == BC_BOTTOM */
    int bot_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[N-1][j] - BC_BOTTOM) > 1e-15) { bot_ok = 0; break; }
    CHECK(bot_ok, "Bottom boundary == BC_BOTTOM");

    /* Left column == BC_LEFT (interior rows; corners belong to top/bottom rows) */
    int left_ok = 1;
    for (int i = 1; i < N - 1; i++)
        if (fabs(g[i][0] - BC_LEFT) > 1e-15) { left_ok = 0; break; }
    CHECK(left_ok, "Left boundary (interior rows) == BC_LEFT");

    /* Right column == BC_RIGHT (interior rows only, same convention) */
    int right_ok = 1;
    for (int i = 1; i < N - 1; i++)
        if (fabs(g[i][N-1] - BC_RIGHT) > 1e-15) { right_ok = 0; break; }
    CHECK(right_ok, "Right boundary (interior rows) == BC_RIGHT");

    /* Interior must be 0.0 after init */
    int interior_ok = 1;
    for (int i = 1; i < N-1; i++)
        for (int j = 1; j < N-1; j++)
            if (fabs(g[i][j]) > 1e-15) { interior_ok = 0; break; }
    CHECK(interior_ok, "Interior is zero-initialised");

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * Test 3 – solver convergence on a small grid
 *   A 10×10 grid with BC_TOP=1 and the other three sides at 0 should
 *   converge well within 10 000 iterations to a tolerance of 1e-5.
 * ------------------------------------------------------------------------- */
static void test_convergence_small(void)
{
    printf("\n[Test 3] Solver convergence (small grid N=10)\n");

    int N = 10;
    double **g = alloc_grid(N);
    init_grid(g, N);

    JacobiResult r = jacobi_solve(g, N, DEFAULT_MAX_ITER, 1e-5);

    CHECK(r.converged,          "Solver reports convergence");
    CHECK(r.iterations > 0,     "At least one iteration was performed");
    CHECK(r.iterations < DEFAULT_MAX_ITER, "Converged before max_iter limit");
    CHECK(r.final_error < 1e-5, "Final error is below requested tolerance");

    /* Boundaries must remain unchanged after the solve */
    int bc_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[0][j] - BC_TOP) > 1e-15) { bc_ok = 0; break; }
    CHECK(bc_ok, "Top boundary (all columns) preserved after solve");

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * Test 4 – solver stops at max_iter when given an unreachable tolerance
 *   Use tol = 0.0 (can never be reached) and a small max_iter cap.
 * ------------------------------------------------------------------------- */
static void test_max_iter_stop(void)
{
    printf("\n[Test 4] Solver stops at max_iter with unreachable tolerance\n");

    int N = 10;
    int max_iter = 5;
    double **g = alloc_grid(N);
    init_grid(g, N);

    JacobiResult r = jacobi_solve(g, N, max_iter, 0.0);

    CHECK(!r.converged,             "Solver did NOT converge (tolerance=0)");
    CHECK(r.iterations == max_iter, "Iteration count equals max_iter cap");

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * Test 5 – discrete Laplace residual check
 *   After convergence the interior must satisfy the discrete Laplacian:
 *     u[i][j] ≈ 0.25*(u[i-1][j]+u[i+1][j]+u[i][j-1]+u[i][j+1])
 *   within a loose tolerance (twice the solver tolerance).
 * ------------------------------------------------------------------------- */
static void test_discrete_laplace_residual(void)
{
    printf("\n[Test 5] Discrete Laplace residual after convergence\n");

    int    N   = 20;
    double tol = 1e-6;
    double **g = alloc_grid(N);
    init_grid(g, N);

    JacobiResult r = jacobi_solve(g, N, DEFAULT_MAX_ITER, tol);
    CHECK(r.converged, "Grid converged (pre-condition for residual test)");

    double max_res = 0.0;
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            double stencil = 0.25 * (g[i-1][j] + g[i+1][j] +
                                     g[i][j-1] + g[i][j+1]);
            double res = fabs(stencil - g[i][j]);
            if (res > max_res) max_res = res;
        }
    }

    /* Allow 2× the solver tolerance as the residual bound */
    CHECK(max_res < 2.0 * tol, "Max discrete Laplace residual < 2×tolerance");

    printf("    (max residual = %.3e, tolerance = %.3e)\n", max_res, tol);

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * Test 6 – solution physically reasonable
 *   With BC_TOP=1 and the other three sides at 0 the entire interior must
 *   stay in the range (0, 1) after convergence.
 * ------------------------------------------------------------------------- */
static void test_solution_range(void)
{
    printf("\n[Test 6] Interior solution stays within boundary value range\n");

    int N = 15;
    double **g = alloc_grid(N);
    init_grid(g, N);
    jacobi_solve(g, N, DEFAULT_MAX_ITER, 1e-6);

    double lo = (BC_TOP < BC_BOTTOM) ? BC_TOP    : BC_BOTTOM;
    double hi = (BC_TOP > BC_BOTTOM) ? BC_TOP    : BC_BOTTOM;
    /* left/right are both 0.0, same as BC_BOTTOM here */

    int range_ok = 1;
    for (int i = 1; i < N-1; i++)
        for (int j = 1; j < N-1; j++)
            if (g[i][j] < lo - 1e-12 || g[i][j] > hi + 1e-12)
                { range_ok = 0; break; }

    CHECK(range_ok, "All interior values lie within [min_BC, max_BC]");

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(void)
{
    printf("=== Serial Jacobi Solver – Unit Tests ===\n");

    test_alloc_free();
    test_boundary_conditions();
    test_convergence_small();
    test_max_iter_stop();
    test_discrete_laplace_residual();
    test_solution_range();

    printf("\n=== Results: %d / %d tests passed ===\n",
           tests_passed, tests_run);

    return (tests_passed == tests_run) ? EXIT_SUCCESS : EXIT_FAILURE;
}
