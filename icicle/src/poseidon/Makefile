test_poseidon: test.cu poseidon.cu kernels.cu constants.cu
	nvcc -o test_poseidon -I../../include -DFIELD_ID=2 -DCURVE_ID=2 test.cu
	./test_poseidon
