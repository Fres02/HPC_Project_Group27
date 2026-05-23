CC        = gcc
NVCC      = nvcc
CFLAGS    = -O2 -Wall -Wextra -std=c11
NVCCFLAGS = -O2 -arch=sm_60
LDFLAGS   = -lm
CUDA_LDFLAGS = -lcudart

CORE_DIR = ../core
CORE_SRCS = $(CORE_DIR)/grid.c $(CORE_DIR)/jacobi_serial.c
CUDA_OBJ = $(CORE_DIR)/jacobi_cuda.o

TEST_BINS = test_grid test_jacobi_serial test_jacobi_cuda

.PHONY: all test clean

all: $(TEST_BINS)

test_grid: test_grid.c $(CORE_DIR)/grid.c
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test_jacobi_serial: test_jacobi_serial.c $(CORE_SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test_jacobi_cuda: test_jacobi_cuda.c $(CORE_DIR)/grid.c $(CUDA_OBJ)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $< $(CORE_DIR)/grid.c $(CUDA_OBJ) $(LDFLAGS) $(CUDA_LDFLAGS)

$(CUDA_OBJ):
	$(MAKE) -C $(CORE_DIR) $(notdir $(CUDA_OBJ))

test: $(TEST_BINS)
	./test_grid
	./test_jacobi_serial
	./test_jacobi_cuda

clean:
	rm -f $(TEST_BINS) *.o
	$(MAKE) -C $(CORE_DIR) clean
