MPICC = mpicc
CFLAGS = -O2 -Wall -Wextra -std=c11 -fopenmp -I../utils
LDFLAGS = -lm -fopenmp

UTILS_LIB = ../utils/libutils.a

.PHONY: all clean

all: jacobi_hybrid.o benchmark_hybrid test_hybrid

jacobi_hybrid.o: jacobi_hybrid.c
	$(MPICC) $(CFLAGS) -c -o $@ $<

benchmark_hybrid: benchmark_hybrid.c jacobi_hybrid.o $(UTILS_LIB)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_hybrid: test_hybrid.c jacobi_hybrid.o $(UTILS_LIB)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o benchmark_hybrid test_hybrid
