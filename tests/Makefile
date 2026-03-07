CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11
LDFLAGS  = -lm

CORE_DIR = ../core
CORE_SRCS = $(CORE_DIR)/grid.c $(CORE_DIR)/jacobi_serial.c

TEST_BINS = test_grid test_jacobi_serial

.PHONY: all test clean

all: $(TEST_BINS)

test_grid: test_grid.c $(CORE_DIR)/grid.c
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test_jacobi_serial: test_jacobi_serial.c $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test: $(TEST_BINS)
	./test_grid
	./test_jacobi_serial

clean:
	rm -f $(TEST_BINS) *.o
