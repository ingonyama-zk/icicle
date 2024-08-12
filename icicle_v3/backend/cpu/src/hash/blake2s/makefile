CC=g++
CFLAGS=-O2 -I. -I../../../include -Wall -Wextra -std=c++11 -pedantic -Wno-long-long
BLAKEBINS=blake2s blake2b blake2sp blake2bp blake2xs blake2xb

blake2s:	blake2s.c
		$(CC) blake2s.c -o $@ $(CFLAGS)
test_blake2s:	test_blake2s.c
		$(CC) test_blake2s.c blake2s.c -o $@ $(CFLAGS)
		./test_blake2s
test_blake2s_integ:	test_blake2s_integ.c
		$(CC) test_blake2s_integ.c blake2s.c -o $@ $(CFLAGS)
		./test_blake2s_integ

clean:
	rm -rf *.o $(BLAKEBINS) test_blake2s test_blake2s_integ