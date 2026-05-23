CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I../utils
LDFLAGS = -lm

UTILS_LIB = ../utils/libutils.a
OBJS = jacobi_serial.o

.PHONY: all clean

all: $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS)
