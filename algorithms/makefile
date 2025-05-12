CC=gcc
CFLAGS=-std=c11 -Wall -O -g 

CFILE=lu.c
OBJECTS=lu.o
BINARY=bin

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CC) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -rf $(BINARY) $(OBJECTS)
