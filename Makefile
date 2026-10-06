CC ?= gcc
CFLAGS ?= -Wall -Wextra -O3 -std=c99
LDFLAGS ?= -lm

SRCS = ztt_ame.c ztt_gemm.c
OBJS = $(SRCS:.c=.o)

TARGET = test_ztt
BENCH = bench_ztt
LIB = libztt.a

.PHONY: all clean test bench riscv

all: $(TARGET)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(TARGET): test_ztt.c $(LIB)
	$(CC) $(CFLAGS) -o $@ test_ztt.c $(LIB) $(LDFLAGS)

$(BENCH): bench_ztt.c ztt_tile_pack.h $(LIB)
	$(CC) $(CFLAGS) -o $@ bench_ztt.c $(LIB) $(LDFLAGS)

%.o: %.c ztt_ame.h ztt_gemm.h
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET)
	./$(TARGET)

bench: $(BENCH)
	./$(BENCH)

riscv:
	riscv64-unknown-linux-gnu-gcc $(CFLAGS) -o test_ztt_riscv test_ztt.c $(SRCS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) $(BENCH) $(LIB) test_ztt_riscv
