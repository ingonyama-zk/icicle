CC=g++
CFLAGS=-O2 -I. -I../../../include -Wall -Wextra -std=c89 -pedantic -Wno-long-long
BLAKEBINS=blake2s blake2b blake2sp blake2bp blake2xs blake2xb

blake2s:	blake2s.c
		$(CC) blake2s.c -o $@ $(CFLAGS) -DBLAKE2S_SELFTEST

clean:
	rm -rf *.o genkat-c genkat-json blake2-kat.h blake2-kat.json $(BLAKEBINS)