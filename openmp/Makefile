CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11 -fopenmp
LDFLAGS  = -lm -fopenmp

CORE_DIR = ../core
CORE_SRCS = $(CORE_DIR)/grid.c $(CORE_DIR)/jacobi_serial.c $(CORE_DIR)/convergence.c

OPENMP_SRCS = jacobi_openmp.c
OPENMP_OBJS = $(OPENMP_SRCS:.c=.o)

BINS = benchmark_openmp test_openmp

.PHONY: all clean

all: $(BINS)

jacobi_openmp.o: jacobi_openmp.c jacobi_openmp.h
	$(CC) $(CFLAGS) -I$(CORE_DIR) -c -o $@ $<

benchmark_openmp: benchmark_openmp.c $(OPENMP_OBJS) $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test_openmp: test_openmp.c $(OPENMP_OBJS) $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BINS) $(OPENMP_OBJS) *.o
