CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

SRCS = grid.c jacobi_serial.c
OBJS = $(SRCS:.c=.o)
DEMO_BIN = demo_solver

.PHONY: all clean demo

all: $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

demo: $(DEMO_BIN)
	./$(DEMO_BIN)

$(DEMO_BIN): demo_solver.c $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(DEMO_BIN)
