CC=g++
CFLAGS=-O2 -I. -I../../../include -I../../../../../include -Wall -Wextra -std=c++17 -pedantic -Wno-long-long


test_keccak:	tests/test_keccak.cpp
		$(CC) tests/test_keccak.cpp keccak.cpp -o $@ $(CFLAGS)
		./test_keccak

test_keccak_sha3: tests/test_keccak_sha3.cpp keccak.cpp 
	$(CC) tests/test_keccak_sha3.cpp keccak.cpp -o $@ $(CFLAGS)
	./test_keccak_sha3

test_keccak256_batch: tests/test_keccak_batch.cpp keccak.cpp 
	$(CC) tests/test_keccak_batch.cpp keccak.cpp -o $@ $(CFLAGS) -DKECCAK256
	./test_keccak256_batch tests/batched_test_vectors_keccak256.csv

test_keccak512_batch: tests/test_keccak_batch.cpp keccak.cpp 
	$(CC) tests/test_keccak_batch.cpp keccak.cpp -o $@ $(CFLAGS) -DKECCAK512
	./test_keccak512_batch tests/batched_test_vectors_keccak512.csv

test_sha3_256_batch: tests/test_keccak_batch.cpp keccak.cpp 
	$(CC) tests/test_keccak_batch.cpp keccak.cpp -o $@ $(CFLAGS) -DSHA3_256
	./test_sha3_256_batch tests/batched_test_vectors_sha3_256.csv

test_sha3_512_batch: tests/test_keccak_batch.cpp keccak.cpp 
	$(CC) tests/test_keccak_batch.cpp keccak.cpp -o $@ $(CFLAGS) -DSHA3_512
	./test_sha3_512_batch tests/batched_test_vectors_sha3_512.csv

clean:
	rm test_keccak_sha3 test_keccak256_batch test_keccak512_batch test_sha3_256_batch test_sha3_512_batch test_keccak