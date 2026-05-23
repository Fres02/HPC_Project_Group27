CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I../utils
LDFLAGS = -lm -lpthread

UTILS_LIB = ../utils/libutils.a

.PHONY: all clean

all: jacobi_pthreads.o

jacobi_pthreads.o: jacobi_pthreads.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o
