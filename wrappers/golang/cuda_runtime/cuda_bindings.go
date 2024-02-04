package cuda_runtime

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
#cgo CFLAGS: -I /usr/local/cuda/include
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

// CudaStreamWaitEvent function as declared in include/cuda_runtime_api.h:2707
func cudaStreamWaitEvent(stream CudaStream, event CudaEvent, flags CudaStreamWaitFlags) CudaError {
	cStream := (C.cudaStream_t)(stream)
	cEvent := (C.cudaEvent_t)(event)
	cFlags := (C.uint)(flags)
	__ret := C.cudaStreamWaitEvent(cStream, cEvent, cFlags)
	__v := (CudaError)(__ret)
	return __v
}

// CudaStreamQuery function as declared in include/cuda_runtime_api.h:2831
func cudaStreamQuery(stream CudaStream) CudaError {
	cStream := (C.cudaStream_t)(stream)
	__ret := C.cudaStreamQuery(cStream)
	__v := (CudaError)(__ret)
	return __v
}

// CudaMemset function as declared in include/cuda_runtime_api.h:7300
func cudaMemset(devPtr unsafe.Pointer, value int32, count uint64) CudaError {
	cValue := (C.int)(value)
	cCount := (C.size_t)(count)
	__ret := C.cudaMemset(devPtr, cValue, cCount)
	__v := (CudaError)(__ret)
	return __v
}

// CudaMemsetAsync function as declared in include/cuda_runtime_api.h:7416
func cudaMemsetAsync(devPtr unsafe.Pointer, value int32, count uint64, stream CudaStream) CudaError {
	cValue := (C.int)(value)
	cCount := (C.size_t)(count)
	cStream := (C.cudaStream_t)(stream)
	__ret := C.cudaMemsetAsync(devPtr, cValue, cCount, cStream)
	__v := (CudaError)(__ret)
	return __v
}
