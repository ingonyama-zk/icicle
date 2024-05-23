test_poseidon: test.cu poseidon2.cu kernels.cu constants.cu
	nvcc -o test_poseidon -I../../include -DFIELD=bn254 -DFIELD_ID=1 -DCURVE_ID=1 -DDEVMODE -DDEBUG extern.cu test.cu
	./test_poseidon

test_poseidon_release: test.cu poseidon2.cu kernels.cu constants.cu
	nvcc -o test_poseidon_release -I../../include -DFIELD=bn254 -DFIELD_ID=1 -DCURVE_ID=1 extern.cu test.cu
	./test_poseidon_release