.PHONY: all clean rebuild structure help

all:
	$(MAKE) -C src all
	$(MAKE) -C tests all

clean:
	$(MAKE) -C src clean
	$(MAKE) -C tests clean
	find . -name "*.o" -delete
	find . -name "*.a" -delete

rebuild: clean all

structure:
	@echo "📁 HPC Jacobi Solver - Project Structure"
	@echo ""
	@echo "src/"
	@echo "├── utils/       - Shared utilities (grid, convergence)"
	@echo "├── serial/      - Serial Jacobi baseline"
	@echo "├── openmp/      - OpenMP parallel (shared memory)"
	@echo "├── pthreads/    - POSIX threads (explicit threading)"
	@echo "├── cuda/        - CUDA GPU acceleration"
	@echo "├── mpi/         - MPI distributed memory"
	@echo "└── hybrid/      - Hybrid MPI+OpenMP combination"
	@echo ""
	@echo "tests/          - Test suite for all implementations"
	@echo "demos/          - Demo applications"
	@echo "docs/           - Documentation and analysis reports"
	@echo ""

help:
	@echo "HPC Jacobi Solver - Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build all implementations and tests"
	@echo "  clean     - Remove all built files and objects"
	@echo "  rebuild   - Clean then build everything"
	@echo "  structure - Show project directory structure"
	@echo "  help      - Display this help message"
	@echo ""
	@echo "Implementation folders can be built individually:"
	@echo "  make -C src/serial       - Build serial version only"
	@echo "  make -C src/openmp       - Build OpenMP version only"
	@echo "  make -C src/pthreads     - Build POSIX threads version only"
	@echo "  make -C src/cuda         - Build CUDA version only"
	@echo "  make -C src/mpi          - Build MPI version only"
	@echo "  make -C src/hybrid       - Build Hybrid (MPI+OpenMP) version only"
