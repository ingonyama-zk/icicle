test_merkle: poseidon2.o merkle.o 
	nvcc -o test_merkle -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -DMERKLE_DEBUG poseidon2.o merkle.o test_poseidon2.cu
	./test_merkle

merkle.o: ../../extern.cu
	nvcc -o merkle.o -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -DMERKLE_DEBUG -c ../../extern.cu

poseidon2.o: ../../../poseidon2/extern.cu
	nvcc -o poseidon2.o -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -c ../../../poseidon2/extern.cu

clear:
	rm -f poseidon2.o merkle.o test_merkle