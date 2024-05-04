test_poseidon: test.cu poseidon.cu kernels.cu constants.cu
	nvcc -o test_poseidon -I../../include -DFIELD=bn254 -DFIELD_ID=1 -DCURVE_ID=1 -DDEVMODE -DDEBUG extern.cu test.cu
	./test_poseidon
