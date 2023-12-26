package cuda_runtime

func GetLastError() CudaError {
	return cudaGetLastError()
}
