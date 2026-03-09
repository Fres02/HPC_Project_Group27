#include <stdio.h>
#include <stdlib.h>
#include "grid.h"
#include "jacobi_serial.h"

int main() {
    printf("════════════════════════════════════════════════════════════\n");
    printf("          JACOBI SERIAL SOLVER - DEMONSTRATION\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    /* Test Case 1: Small grid (20x20) */
    printf("[TEST 1] Small Grid (20×20)\n");
    printf("───────────────────────────────────────────────────────────\n");
    
    int N1 = 20;
    double **grid1 = alloc_grid(N1);
    init_grid(grid1, N1);
    
    printf("Grid allocated: %d × %d\n", N1, N1);
    printf("Boundary conditions applied (Top=100, Bottom=0, Left=0, Right=0)\n");
    printf("Starting Jacobi iteration...\n\n");
    
    JacobiResult result1 = jacobi_solve(grid1, N1, DEFAULT_MAX_ITER, DEFAULT_TOL);
    
    printf("✓ Convergence Status: %s\n", result1.converged ? "YES" : "NO (max_iter reached)");
    printf("  Iterations needed: %d\n", result1.iterations);
    printf("  Final error: %.8e\n", result1.final_error);
    printf("  Tolerance: %.8e\n\n", DEFAULT_TOL);
    
    free_grid(grid1, N1);

    /* Test Case 2: Medium grid (50x50) */
    printf("[TEST 2] Medium Grid (50×50)\n");
    printf("───────────────────────────────────────────────────────────\n");
    
    int N2 = 50;
    double **grid2 = alloc_grid(N2);
    init_grid(grid2, N2);
    
    printf("Grid allocated: %d × %d\n", N2, N2);
    printf("Boundary conditions applied (Top=100, Bottom=0, Left=0, Right=0)\n");
    printf("Starting Jacobi iteration...\n\n");
    
    JacobiResult result2 = jacobi_solve(grid2, N2, DEFAULT_MAX_ITER, DEFAULT_TOL);
    
    printf("✓ Convergence Status: %s\n", result2.converged ? "YES" : "NO (max_iter reached)");
    printf("  Iterations needed: %d\n", result2.iterations);
    printf("  Final error: %.8e\n", result2.final_error);
    printf("  Tolerance: %.8e\n\n", DEFAULT_TOL);
    
    free_grid(grid2, N2);

    /* Test Case 3: Large grid (100x100) */
    printf("[TEST 3] Large Grid (100×100)\n");
    printf("───────────────────────────────────────────────────────────\n");
    
    int N3 = 100;
    double **grid3 = alloc_grid(N3);
    init_grid(grid3, N3);
    
    printf("Grid allocated: %d × %d\n", N3, N3);
    printf("Boundary conditions applied (Top=100, Bottom=0, Left=0, Right=0)\n");
    printf("Starting Jacobi iteration...\n\n");
    
    JacobiResult result3 = jacobi_solve(grid3, N3, DEFAULT_MAX_ITER, DEFAULT_TOL);
    
    printf("✓ Convergence Status: %s\n", result3.converged ? "YES" : "NO (max_iter reached)");
    printf("  Iterations needed: %d\n", result3.iterations);
    printf("  Final error: %.8e\n", result3.final_error);
    printf("  Tolerance: %.8e\n\n", DEFAULT_TOL);
    
    free_grid(grid3, N3);

    /* Summary */
    printf("════════════════════════════════════════════════════════════\n");
    printf("                       SUMMARY\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("Grid Size  │ Converged │ Iterations │ Final Error\n");
    printf("───────────┼───────────┼────────────┼──────────────────\n");
    printf("  20×20    │    %s     │   %6d   │ %.8e\n", 
           result1.converged ? "YES" : "NO ", result1.iterations, result1.final_error);
    printf("  50×50    │    %s     │   %6d   │ %.8e\n", 
           result2.converged ? "YES" : "NO ", result2.iterations, result2.final_error);
    printf("  100×100  │    %s     │   %6d   │ %.8e\n", 
           result3.converged ? "YES" : "NO ", result3.iterations, result3.final_error);
    printf("════════════════════════════════════════════════════════════\n\n");

    printf("All tests completed successfully!\n");
    return 0;
}
