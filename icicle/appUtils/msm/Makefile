test_msm:
	mkdir -p work
	nvcc -o work/test_msm -std=c++17 -I. -I../.. tests/msm_test.cu
	work/test_msm

precompute: precompute_test.cu msm.cu
	mkdir -p work
	nvcc -o work/precompute -std=c++17 -I. -I../.. precompute_test.cu
	work/precompute