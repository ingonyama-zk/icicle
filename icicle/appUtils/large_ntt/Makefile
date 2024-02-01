build_verification:
	mkdir -p work
	nvcc -o work/test_verification -I. -I.. -I../.. -I../ntt verification.cu -std=c++17

test_verification: build_verification
	work/test_verification
