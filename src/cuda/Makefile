NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_60 -I../utils

UTILS_LIB = ../utils/libutils.a

.PHONY: all clean

all: jacobi_cuda.o

jacobi_cuda.o: jacobi_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f *.o
