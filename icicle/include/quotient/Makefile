build_quotient:
	mkdir -p work
	nvcc -o work/test_msm -std=c++17 -arch=sm_80 -I. -I../../include tests/quotient_test.cu

test_quotient:
	mkdir -p work
	nvcc -o work/test_msm -std=c++17 -arch=sm_80 -I.  -I../../include tests/quotient_test.cu
	work/test_msm