test_merkle: poseidon2.o mmcs.o vec_ops.o
	nvcc -o test_mmcs_poseidon2 -lineinfo -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -DMERKLE_DEBUG poseidon2.o vec_ops.o mmcs.o test_poseidon2.cu
	./test_mmcs_poseidon2

mmcs.o: ../../extern_mmcs.cu ../../mmcs.cu
	nvcc -o mmcs.o -I../../../../include -lineinfo -DFIELD=babybear -DFIELD_ID=1001 -DMERKLE_DEBUG -c ../../extern_mmcs.cu

poseidon2.o: ../../../poseidon2/extern.cu
	nvcc -o poseidon2.o -I../../../../include -lineinfo -DFIELD=babybear -DFIELD_ID=1001 -c ../../../poseidon2/extern.cu

vec_ops.o:
	nvcc -o vec_ops.o -I../../../../include -lineinfo -DFIELD=babybear -DFIELD_ID=1001 -c ../../../vec_ops/extern.cu

clear:
	rm -f poseidon2.o mmcs.o vec_ops.o test_mmcs_poseidon2

build_merkle_blake:
	nvcc -o test_mmcs_blake2s -lineinfo -I../../../../include -DMERKLE_DEBUG ../../../hash/blake2s/extern.cu test_blake2s.cu