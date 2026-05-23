CC       = gcc
NVCC     = nvcc
CFLAGS   = -O2 -Wall -Wextra -std=c11
NVCCFLAGS = -O2 -arch=sm_60
LDFLAGS  = -lm

SRCS = grid.c jacobi_serial.c
CUDA_SRCS = jacobi_cuda.cu
OBJS = $(SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
DEMO_BIN = demo_solver

.PHONY: all clean demo

all: $(OBJS) $(CUDA_OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

demo: $(DEMO_BIN)
	./$(DEMO_BIN)

$(DEMO_BIN): demo_solver.c $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(CUDA_OBJS) $(DEMO_BIN)
