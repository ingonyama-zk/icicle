test_poseidon: test.cu 
	nvcc -o test_poseidon -I../../include -DFIELD=bls12_381 -DFIELD_ID=2 -DCURVE_ID=2 -DDEVMODE -DDEBUG extern.cu test.cu
	./test_poseidon