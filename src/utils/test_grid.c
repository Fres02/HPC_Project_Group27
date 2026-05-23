/*
 * test_grid.c
 *
 * Unit tests for Feature 2: Grid and Boundary Management (core/grid.c).
 *
 * Build (from tests/ directory):
 *   make
 *
 * Run:
 *   ./test_grid
 */

#include "grid.h"

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
 *   Verify: non-NULL return, N×N contiguous row-major layout, crash-free free.
 * ------------------------------------------------------------------------- */
static void test_alloc_free(void)
{
    printf("\n[Test 1] alloc_grid / free_grid (N=10)\n");

    int N = 10;
    double **g = alloc_grid(N);
    CHECK(g != NULL, "alloc_grid returns non-NULL");

    /* Row pointers must be exactly N doubles apart (contiguous) */
    int contiguous = 1;
    for (int i = 1; i < N; i++)
        if (g[i] != g[0] + (size_t)i * N) { contiguous = 0; break; }
    CHECK(contiguous, "Row pointers are contiguous (row-major layout)");

    free_grid(g, N);
    CHECK(1, "free_grid completes without crash");
}

/* -------------------------------------------------------------------------
 * Test 2 – alloc_grid edge case N=1
 *   A single-cell grid must still allocate successfully.
 * ------------------------------------------------------------------------- */
static void test_alloc_single_cell(void)
{
    printf("\n[Test 2] alloc_grid edge case N=1\n");

    double **g = alloc_grid(1);
    CHECK(g != NULL, "alloc_grid(1) returns non-NULL");
    if (g) {
        g[0][0] = 42.0;
        CHECK(fabs(g[0][0] - 42.0) < 1e-15, "Single cell is writable");
        free_grid(g, 1);
    }
}

/* -------------------------------------------------------------------------
 * Test 3 – free_grid on NULL pointer
 *   free_grid(NULL, N) must not crash (graceful no-op).
 * ------------------------------------------------------------------------- */
static void test_free_null(void)
{
    printf("\n[Test 3] free_grid(NULL) is a no-op\n");
    free_grid(NULL, 5);
    CHECK(1, "free_grid(NULL) does not crash");
}

/* -------------------------------------------------------------------------
 * Test 4 – init_grid default boundary conditions
 *   After init_grid the four edges must match BC_* constants and the
 *   interior must be 0.0.
 * ------------------------------------------------------------------------- */
static void test_init_grid_default(void)
{
    printf("\n[Test 4] init_grid default boundary conditions (N=8)\n");

    int N = 8;
    double **g = alloc_grid(N);
    init_grid(g, N);

    int top_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[0][j] - BC_TOP) > 1e-15) { top_ok = 0; break; }
    CHECK(top_ok, "Top row == BC_TOP (full width including corners)");

    int bot_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[N-1][j] - BC_BOTTOM) > 1e-15) { bot_ok = 0; break; }
    CHECK(bot_ok, "Bottom row == BC_BOTTOM (full width including corners)");

    int left_ok = 1;
    for (int i = 1; i < N - 1; i++)
        if (fabs(g[i][0] - BC_LEFT) > 1e-15) { left_ok = 0; break; }
    CHECK(left_ok, "Left column == BC_LEFT (interior rows)");

    int right_ok = 1;
    for (int i = 1; i < N - 1; i++)
        if (fabs(g[i][N-1] - BC_RIGHT) > 1e-15) { right_ok = 0; break; }
    CHECK(right_ok, "Right column == BC_RIGHT (interior rows)");

    int interior_ok = 1;
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            if (fabs(g[i][j]) > 1e-15) { interior_ok = 0; break; }
    CHECK(interior_ok, "Interior is zero-initialised");

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * Test 5 – init_grid_custom with non-default values
 *   Custom BCs and interior fill should be applied independently.
 * ------------------------------------------------------------------------- */
static void test_init_grid_custom(void)
{
    printf("\n[Test 5] init_grid_custom with user-supplied values (N=6)\n");

    int N = 6;
    double top = 5.0, bottom = -3.0, left = 2.5, right = -1.0, fill = 7.0;
    double **g = alloc_grid(N);
    init_grid_custom(g, N, top, bottom, left, right, fill);

    int top_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[0][j] - top) > 1e-15) { top_ok = 0; break; }
    CHECK(top_ok, "Top row matches custom top value");

    int bot_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[N-1][j] - bottom) > 1e-15) { bot_ok = 0; break; }
    CHECK(bot_ok, "Bottom row matches custom bottom value");

    int left_ok = 1;
    for (int i = 1; i < N - 1; i++)
        if (fabs(g[i][0] - left) > 1e-15) { left_ok = 0; break; }
    CHECK(left_ok, "Left column matches custom left value (interior rows)");

    int right_ok = 1;
    for (int i = 1; i < N - 1; i++)
        if (fabs(g[i][N-1] - right) > 1e-15) { right_ok = 0; break; }
    CHECK(right_ok, "Right column matches custom right value (interior rows)");

    int interior_ok = 1;
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            if (fabs(g[i][j] - fill) > 1e-15) { interior_ok = 0; break; }
    CHECK(interior_ok, "Interior filled with custom interior_val");

    /* Corners must belong to top/bottom rows, not left/right columns */
    CHECK(fabs(g[0][0]     - top)    < 1e-15, "Top-left corner == top value");
    CHECK(fabs(g[0][N-1]   - top)    < 1e-15, "Top-right corner == top value");
    CHECK(fabs(g[N-1][0]   - bottom) < 1e-15, "Bottom-left corner == bottom value");
    CHECK(fabs(g[N-1][N-1] - bottom) < 1e-15, "Bottom-right corner == bottom value");

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * Test 6 – fill_interior does not touch boundary
 *   After init_grid, fill_interior with a new value; the boundary should
 *   remain unchanged while all interior points take the new value.
 * ------------------------------------------------------------------------- */
static void test_fill_interior(void)
{
    printf("\n[Test 6] fill_interior preserves boundary (N=5)\n");

    int N = 5;
    double **g = alloc_grid(N);
    init_grid(g, N);      /* boundary set, interior = 0 */
    fill_interior(g, N, 99.0);

    int interior_ok = 1;
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            if (fabs(g[i][j] - 99.0) > 1e-15) { interior_ok = 0; break; }
    CHECK(interior_ok, "Interior cells updated to new value");

    /* Boundary must not have been overwritten */
    int top_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[0][j] - BC_TOP) > 1e-15) { top_ok = 0; break; }
    CHECK(top_ok, "Top boundary unchanged after fill_interior");

    int bot_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[N-1][j] - BC_BOTTOM) > 1e-15) { bot_ok = 0; break; }
    CHECK(bot_ok, "Bottom boundary unchanged after fill_interior");

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * Test 7 – swap_buffers exchanges pointer arrays
 *   Write distinct marker values into two grids, swap, verify the pointers
 *   were exchanged and the underlying data is intact.
 * ------------------------------------------------------------------------- */
static void test_swap_buffers(void)
{
    printf("\n[Test 7] swap_buffers exchanges grid pointers (N=4)\n");

    int N = 4;
    double **a = alloc_grid(N);
    double **b = alloc_grid(N);

    fill_interior(a, N, 1.0);
    fill_interior(b, N, 2.0);

    double **orig_a = a;
    double **orig_b = b;

    swap_buffers(&a, &b);

    CHECK(a == orig_b, "After swap: a points to original b allocation");
    CHECK(b == orig_a, "After swap: b points to original a allocation");

    /* Data in the buffers must be unchanged – only pointers moved */
    CHECK(fabs(a[2][2] - 2.0) < 1e-15, "Data accessed via new a is former b's data");
    CHECK(fabs(b[2][2] - 1.0) < 1e-15, "Data accessed via new b is former a's data");

    /* Swap back – must be idempotent */
    swap_buffers(&a, &b);
    CHECK(a == orig_a, "Double-swap restores original a pointer");
    CHECK(b == orig_b, "Double-swap restores original b pointer");

    free_grid(a, N);
    free_grid(b, N);
}

/* -------------------------------------------------------------------------
 * Test 8 – copy_grid produces an independent copy
 *   Modifying the source after copying must not affect the destination.
 * ------------------------------------------------------------------------- */
static void test_copy_grid(void)
{
    printf("\n[Test 8] copy_grid produces independent copy (N=6)\n");

    int N = 6;
    double **src = alloc_grid(N);
    double **dst = alloc_grid(N);

    init_grid(src, N);
    fill_interior(src, N, 3.14);

    copy_grid(dst, src, N);

    /* Spot-check: dst content matches src */
    int match = 1;
    for (int i = 0; i < N && match; i++)
        for (int j = 0; j < N && match; j++)
            if (fabs(dst[i][j] - src[i][j]) > 1e-15) match = 0;
    CHECK(match, "dst matches src immediately after copy");

    /* Mutate src; dst must remain unchanged */
    fill_interior(src, N, -99.0);
    int independent = 1;
    for (int i = 1; i < N - 1 && independent; i++)
        for (int j = 1; j < N - 1 && independent; j++)
            if (fabs(dst[i][j] - 3.14) > 1e-15) independent = 0;
    CHECK(independent, "dst is independent of src after mutation");

    free_grid(src, N);
    free_grid(dst, N);
}

/* -------------------------------------------------------------------------
 * Test 9 – init_grid is idempotent
 *   Calling init_grid twice on the same grid must produce the same result
 *   as calling it once (no state leakage between calls).
 * ------------------------------------------------------------------------- */
static void test_init_grid_idempotent(void)
{
    printf("\n[Test 9] init_grid is idempotent (N=7)\n");

    int N = 7;
    double **g = alloc_grid(N);
    init_grid(g, N);
    fill_interior(g, N, 99.0); /* dirty the interior */
    init_grid(g, N);           /* re-initialise      */

    int interior_ok = 1;
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            if (fabs(g[i][j]) > 1e-15) { interior_ok = 0; break; }
    CHECK(interior_ok, "Interior is 0 after second init_grid");

    int top_ok = 1;
    for (int j = 0; j < N; j++)
        if (fabs(g[0][j] - BC_TOP) > 1e-15) { top_ok = 0; break; }
    CHECK(top_ok, "Boundary still correct after second init_grid");

    free_grid(g, N);
}

/* -------------------------------------------------------------------------
 * Test 10 – large grid allocation
 *   Verify the module handles a larger grid (N=512) without errors.
 * ------------------------------------------------------------------------- */
static void test_large_alloc(void)
{
    printf("\n[Test 10] Large grid allocation (N=512)\n");

    int N = 512;
    double **g = alloc_grid(N);
    CHECK(g != NULL, "alloc_grid(512) returns non-NULL");
    if (g) {
        init_grid(g, N);
        CHECK(fabs(g[0][N/2] - BC_TOP) < 1e-15,
              "Boundary value readable on large grid");
        free_grid(g, N);
        CHECK(1, "free_grid on large grid does not crash");
    }
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(void)
{
    printf("=== Feature 2: Grid and Boundary Management – Unit Tests ===\n");

    test_alloc_free();
    test_alloc_single_cell();
    test_free_null();
    test_init_grid_default();
    test_init_grid_custom();
    test_fill_interior();
    test_swap_buffers();
    test_copy_grid();
    test_init_grid_idempotent();
    test_large_alloc();

    printf("\n=== Results: %d / %d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
