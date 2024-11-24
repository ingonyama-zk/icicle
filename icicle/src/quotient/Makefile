build_quotient:
	mkdir -p work
	nvcc -g -G -o work/quotient_test -std=c++17 -arch=sm_80 -I. -I../../include tests/quotient_test.cu

test_quotient:
	mkdir -p work
	nvcc -g -G -o work/quotient_test -std=c++17 -arch=sm_80 -I.  -I../../include tests/quotient_test.cu
	work/quotient_test