CC=g++
CFLAGS=-O2 -I. -I../../../include -I../../../../../include -Wall -Wextra -std=c++17 -pedantic -Wno-long-long
BLAKEBINS=blake2s blake2b blake2sp blake2bp blake2xs blake2xb

test_blake2s:	tests/test_blake2s.c
		$(CC) tests/test_blake2s.c blake2s.c -o $@ $(CFLAGS)
		./test_blake2s
test_blake2s_integ:	tests/test_blake2s_integ.c
		$(CC) tests/test_blake2s_integ.c blake2s.c -o $@ $(CFLAGS)
		./test_blake2s_integ
test_blake2s_seq:	tests/test_blake2s_seq.c
		$(CC) tests/test_blake2s_seq.c blake2s.c -o $@ $(CFLAGS)
		./test_blake2s_seq
test_blake2s_batch:	tests/test_blake2s_batch.c
		$(CC) tests/test_blake2s_batch.c blake2s.c -o $@ $(CFLAGS)
		./test_blake2s_batch tests/batched_test_vectors.csv

clean:
	rm -rf *.o $(BLAKEBINS) test_blake2s test_blake2s_integ test_blake2s_seq test_blake2s_batch