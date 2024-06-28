test_merkle: poseidon2.o mmcs.o 
	nvcc -o test_mmcs_poseidon2 -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -DMERKLE_DEBUG poseidon2.o mmcs.o test_poseidon2.cu
	./test_mmcs_poseidon2

mmcs.o: ../../extern_mmcs.cu
	nvcc -o mmcs.o -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -DMERKLE_DEBUG -c ../../extern_mmcs.cu

poseidon2.o: ../../../poseidon2/extern.cu
	nvcc -o poseidon2.o -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -c ../../../poseidon2/extern.cu

clear:
	rm -f poseidon2.o mmcs.o test_mmcs_poseidon2