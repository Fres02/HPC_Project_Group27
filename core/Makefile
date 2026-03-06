CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

SRCS    = jacobi_serial.c
OBJS    = $(SRCS:.c=.o)

TEST_SRC = test_jacobi_serial.c
TEST_BIN = test_jacobi_serial

.PHONY: all test clean

all: $(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(TEST_BIN) *.o
