test_merkle_poseidon: poseidon.o merkle_bls.o 
	nvcc -o test_merkle_poseidon -I../../../../include -DFIELD=bls12_381 -DFIELD_ID=2 -DCURVE=bls12_381 -DMERKLE_DEBUG poseidon.o merkle_bls.o test.cu
	./test_merkle_poseidon

merkle_bls.o: ../../extern.cu ../../merkle.cu
	nvcc -o merkle_bls.o -I../../../../include -DFIELD=bls12_381 -DFIELD_ID=2 -DCURVE=bls12_381 -DMERKLE_DEBUG -c ../../extern.cu

poseidon.o: ../../../poseidon/extern.cu
	nvcc -o poseidon.o -I../../../../include -DFIELD=bls12_381 -DFIELD_ID=2 -DCURVE=bls12_381 -c ../../../poseidon/extern.cu


test_merkle: poseidon2.o merkle.o 
	nvcc -o test_merkle -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -DMERKLE_DEBUG poseidon2.o merkle.o test_poseidon2.cu
	./test_merkle

merkle.o: ../../extern.cu ../../merkle.cu
	nvcc -o merkle.o -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -DMERKLE_DEBUG -c ../../extern.cu

poseidon2.o: ../../../poseidon2/extern.cu
	nvcc -o poseidon2.o -I../../../../include -DFIELD=babybear -DFIELD_ID=1001 -c ../../../poseidon2/extern.cu

clear:
	rm -f poseidon2.o merkle.o test_merkle merkle_bls.o poseidon.o test_merkle