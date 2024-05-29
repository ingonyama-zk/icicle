test_merkle:
	nvcc -o test_merkle -I../../include -DFIELD=bls12_381 -DFIELD_ID=2 -DCURVE_ID=2 -DMERKLE_DEBUG ../poseidon/extern.cu extern_poseidon.cu test.cu
	./test_merkle