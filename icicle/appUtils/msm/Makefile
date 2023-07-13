test_msm:
	mkdir -p work
	nvcc -o work/test_msm -I. tests/msm_test.cu
	work/test_msm

test_msm_debug:
	mkdir -p work
	nvcc -o work/test_msm_debug -I. tests/msm_test.cu -g -G
	work/test_msm_debug