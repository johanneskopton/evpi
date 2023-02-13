CC = gcc
CFLAGS = -Wall -std=c99
MYDIR=build

all: main

main: mkdir read_csv.o evpi.o
	$(CC) $(CFLAGS) -o build/read_csv build/read_csv.o build/evpi.o

mkdir:
	[ -d $(MYDIR) ] || mkdir -p $(MYDIR)

read_csv.o: read_csv.c
	$(CC) $(CFLAGS) -c -o build/read_csv.o read_csv.c

evpi.o: evpi.c
	$(CC) $(CFLAGS) -c -o build/evpi.o evpi.c

clean:
	rm -r ./build