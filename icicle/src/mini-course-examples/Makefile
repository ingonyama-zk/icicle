build_test:
	mkdir -p work
	nvcc -o work/test -std=c++17 -arch=sm_80 -I. -I../../include test.cu

run_test:
	mkdir -p work
	nvcc -o work/test -std=c++17 -arch=sm_80 -I.  -I../../include test.cu
	work/test