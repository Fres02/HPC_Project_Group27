CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11
LDFLAGS  = -lm

CORE_DIR = ../core
SRCS     = $(CORE_DIR)/grid.c
TEST_SRC = test_grid.c
TEST_BIN = test_grid

.PHONY: all test clean

all: $(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(SRCS)
	$(CC) $(CFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(TEST_BIN) *.o
