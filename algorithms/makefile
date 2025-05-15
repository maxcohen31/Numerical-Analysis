CC=gcc
CFLAGS=-std=c11 -Wall -O -g -pthread 

CFILE=multi_thread_lu.c
OBJECTS=multi_thread_lu.o
BINARY=bin

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CC) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -rf $(BINARY) $(OBJECTS)
