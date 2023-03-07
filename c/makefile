CC = gcc
CFLAGS = -Wall -std=c99
MYDIR=build

all: main

main: mkdir test.o evpi.o
	$(CC) $(CFLAGS) -o build/test build/test.o build/evpi.o -lm

mkdir:
	[ -d $(MYDIR) ] || mkdir -p $(MYDIR)

test.o: test.c
	$(CC) $(CFLAGS) -c -o build/test.o test.c

evpi.o: evpi.c
	$(CC) $(CFLAGS) -c -o build/evpi.o evpi.c

clean:
	rm -r ./build