build_test:
	mkdir -p work
	nvcc -o work/test -std=c++17 -arch=sm_80 -I. -I../../include test.cu

run_test:
	mkdir -p work
	nvcc -o work/test -std=c++17 -arch=sm_80 -I.  -I../../include test.cu
	work/test


build_perf:
	mkdir -p work
	nvcc -lineinfo -o work/perf -std=c++17 -arch=sm_80 -I. -I../../include perf_test.cu

run_perf:
	make build_perf
	work/perf


build_mem:
	mkdir -p work
	nvcc -lineinfo -o work/mem -std=c++17 -arch=sm_80 -I. -I../../include memory_test.cu

run_mem:
	make build_mem
	work/mem