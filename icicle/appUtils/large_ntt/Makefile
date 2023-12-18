test_debug_ntt:
	mkdir -p work
	nvcc --generate-line-info -o work/test_ntt -I. Tester.cu
	work/test_ntt 1 10

test_ntt:
	mkdir -p work
	nvcc -o work/test_ntt -I. Tester.cu
	work/test_ntt 1 1