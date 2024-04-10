build_verification:
	mkdir -p work
	nvcc -o work/test_verification -I. -I../../include tests/verification.cu -std=c++17

test_verification: build_verification
	work/test_verification

build_verification_ecntt:
	mkdir -p work
	nvcc -o work/test_verification_ecntt -I. -I../../include tests/verification.cu -std=c++17 -DECNTT

test_verification_ecntt: build_verification_ecntt
	work/test_verification_ecntt
