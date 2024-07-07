test_poseidon: test.cu 
	nvcc -o test_poseidon -I../../include -DFIELD=bn254 -DFIELD_ID=1 -DCURVE_ID=1 extern.cu test.cu