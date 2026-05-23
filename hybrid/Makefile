CC       = mpicc
CFLAGS   = -O2 -Wall -Wextra -std=c11 -fopenmp
LDFLAGS  = -lm -fopenmp

CORE_DIR  = ../core
CORE_SRCS = $(CORE_DIR)/grid.c $(CORE_DIR)/jacobi_serial.c $(CORE_DIR)/convergence.c

HYB_SRCS  = jacobi_hybrid.c
HYB_OBJS  = $(HYB_SRCS:.c=.o)

BINS = benchmark_hybrid test_hybrid

.PHONY: all clean

all: $(BINS)

jacobi_hybrid.o: jacobi_hybrid.c jacobi_hybrid.h
	$(CC) $(CFLAGS) -I$(CORE_DIR) -c -o $@ $<

benchmark_hybrid: benchmark_hybrid.c $(HYB_OBJS) $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test_hybrid: test_hybrid.c $(HYB_OBJS) $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BINS) $(HYB_OBJS) *.o *.exe
