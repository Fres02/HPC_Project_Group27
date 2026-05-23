CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11

SRCS = grid.c convergence.c
OBJS = $(SRCS:.c=.o)
LIB = libutils.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(LIB)
