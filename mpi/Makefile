CC       = mpicc
CFLAGS   = -O2 -Wall -Wextra -std=c11
LDFLAGS  = -lm

CORE_DIR  = ../core
CORE_SRCS = $(CORE_DIR)/grid.c $(CORE_DIR)/jacobi_serial.c $(CORE_DIR)/convergence.c

MPI_SRCS  = jacobi_mpi.c
MPI_OBJS  = $(MPI_SRCS:.c=.o)

BINS = benchmark_mpi test_mpi

.PHONY: all clean

all: $(BINS)

jacobi_mpi.o: jacobi_mpi.c jacobi_mpi.h
	$(CC) $(CFLAGS) -I$(CORE_DIR) -c -o $@ $<

benchmark_mpi: benchmark_mpi.c $(MPI_OBJS) $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test_mpi: test_mpi.c $(MPI_OBJS) $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BINS) $(MPI_OBJS) *.o *.exe
