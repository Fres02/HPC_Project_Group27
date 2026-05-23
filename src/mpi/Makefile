MPICC = mpicc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I../utils
LDFLAGS = -lm

UTILS_LIB = ../utils/libutils.a

.PHONY: all clean

all: jacobi_mpi.o benchmark_mpi test_mpi

jacobi_mpi.o: jacobi_mpi.c
	$(MPICC) $(CFLAGS) -c -o $@ $<

benchmark_mpi: benchmark_mpi.c jacobi_mpi.o $(UTILS_LIB)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_mpi: test_mpi.c jacobi_mpi.o $(UTILS_LIB)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o benchmark_mpi test_mpi
