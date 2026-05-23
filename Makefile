CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

CORE_DIR   = core
TESTS_DIR  = tests
OPENMP_DIR = openmp

CORE_SRCS = $(CORE_DIR)/grid.c $(CORE_DIR)/jacobi_serial.c

TEST_BINS = $(TESTS_DIR)/test_grid $(TESTS_DIR)/test_jacobi_serial

.PHONY: all test clean openmp

all: $(TEST_BINS)

$(TESTS_DIR)/test_grid: $(TESTS_DIR)/test_grid.c $(CORE_DIR)/grid.c
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

$(TESTS_DIR)/test_jacobi_serial: $(TESTS_DIR)/test_jacobi_serial.c $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test: $(TEST_BINS)
	$(TESTS_DIR)/test_grid
	$(TESTS_DIR)/test_jacobi_serial

openmp:
	@echo "Building OpenMP implementation..."
	$(MAKE) -C $(OPENMP_DIR)

clean:
	rm -f $(TEST_BINS)
	$(MAKE) -C $(OPENMP_DIR) clean
