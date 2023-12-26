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

// CudaGetLastError function as declared in include/cuda_runtime_api.h:1305
func cudaGetLastError() CudaError {
	__ret := C.cudaGetLastError()
	__v := (CudaError)(__ret)
	return __v
}

// CudaSetDevice function as declared in include/cuda_runtime_api.h:2185
func cudaSetDevice(device int32) CudaError {
	cDevice := (C.int)(device)
	__ret := C.cudaSetDevice(cDevice)
	__v := (CudaError)(__ret)
	return __v
}

// CudaStreamCreate function as declared in include/cuda_runtime_api.h:2393
func cudaStreamCreate(pStream *CudaStream) CudaError {
	cPStream := (*C.cudaStream_t)(pStream)
	__ret := C.cudaStreamCreate(cPStream)
	__v := (CudaError)(__ret)
	return __v
}

// CudaStreamCreateWithFlags function as declared in include/cuda_runtime_api.h:2425
func cudaStreamCreateWithFlags(pStream *CudaStream, flags CudaStreamCreateFlags) CudaError {
	cPStream := (*C.cudaStream_t)(pStream)
	cFlags := (C.uint)(flags)
	__ret := C.cudaStreamCreateWithFlags(cPStream, cFlags)
	__v := (CudaError)(__ret)
	return __v
}

// CudaStreamDestroy function as declared in include/cuda_runtime_api.h:2676
func cudaStreamDestroy(stream CudaStream) CudaError {
	cStream := (C.cudaStream_t)(stream)
	__ret := C.cudaStreamDestroy(cStream)
	__v := (CudaError)(__ret)
	return __v
}

// CudaStreamWaitEvent function as declared in include/cuda_runtime_api.h:2707
func cudaStreamWaitEvent(stream CudaStream, event CudaEvent, flags CudaStreamWaitFlags) CudaError {
	cStream := (C.cudaStream_t)(stream)
	cEvent := (C.cudaEvent_t)(event)
	cFlags := (C.uint)(flags)
	__ret := C.cudaStreamWaitEvent(cStream, cEvent, cFlags)
	__v := (CudaError)(__ret)
	return __v
}

// CudaStreamSynchronize function as declared in include/cuda_runtime_api.h:2806
func cudaStreamSynchronize(stream CudaStream) CudaError {
	cStream := (C.cudaStream_t)(stream)
	__ret := C.cudaStreamSynchronize(cStream)
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

// CudaMallocAsync function as declared in include/cuda_runtime_api.h:8314
func cudaMallocAsync(devPtr unsafe.Pointer, size uint64, stream CudaStream) CudaError {
	cSize := (C.size_t)(size)
	cStream := (C.cudaStream_t)(stream)
	__ret := C.cudaMallocAsync(&devPtr, cSize, cStream)
	__v := (CudaError)(__ret)
	return __v
}

// CudaMalloc function as declared in include/cuda_runtime_api.h:5207
func cudaMalloc(devPtr unsafe.Pointer, size uint64) CudaError {
	cSize := (C.size_t)(size)
	__ret := C.cudaMalloc(&devPtr, cSize)
	__v := (CudaError)(__ret)
	return __v
}

// CudaFreeAsync function as declared in include/cuda_runtime_api.h:8340
func cudaFreeAsync(devPtr unsafe.Pointer, stream CudaStream) CudaError {
	cStream := (C.cudaStream_t)(stream)
	__ret := C.cudaFreeAsync(devPtr, cStream)
	__v := (CudaError)(__ret)
	return __v
}

// CudaFree function as declared in include/cuda_runtime_api.h:5377
func cudaFree(devPtr unsafe.Pointer) CudaError {
	__ret := C.cudaFree(devPtr)
	__v := (CudaError)(__ret)
	return __v
}

func cudaMemcpy(dst, src unsafe.Pointer, count uint64, kind CudaMemcpyKind) CudaError {
	cCount := (C.size_t)(count)
	__ret := C.cudaMemcpy(dst, src, cCount, uint32(kind))
	__v := (CudaError)(__ret)
	return __v
}

func cudaMemcpyAsync(dst, src unsafe.Pointer, count int, kind CudaMemcpyKind, stream CudaStream) CudaError {
	cCount := (C.size_t)(count)
	cStream := (C.cudaStream_t)(stream)
	__ret := C.cudaMemcpyAsync(dst, src, cCount, uint32(kind), cStream)
	__v := (CudaError)(__ret)
	return __v
}
