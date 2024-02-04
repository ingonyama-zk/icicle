package cuda_runtime

// #cgo CFLAGS: -I /usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

import (
	"unsafe"
)

type HostOrDeviceSlice[T, S any] interface {
	Len() int
	Cap() int
	IsEmpty() bool
	AsSlice() T
	AsPointer() *S
	IsOnDevice() bool
}

func Malloc(size uint) (unsafe.Pointer, CudaError) {
	if size == 0 {
		return nil, CudaErrorMemoryAllocation
	}

	var p C.void
	devicePtr := unsafe.Pointer(&p)
	cSize := (C.size_t)(size)
	
	ret := C.cudaMalloc(&devicePtr, cSize)
	err := (CudaError)(ret)

	return devicePtr, err
}

func MallocAsync(size uint, stream CudaStream) (unsafe.Pointer, CudaError) {
	if size == 0 {
		return nil, CudaErrorMemoryAllocation
	}

	var p C.void
	devicePtr := unsafe.Pointer(&p)
	cSize := (C.size_t)(size)
	cStream := (C.cudaStream_t)(stream)

	ret := C.cudaMallocAsync(&devicePtr, cSize, cStream)
	err := (CudaError)(ret)

	return devicePtr, err
}

func Free(devicePtr unsafe.Pointer) CudaError {
	ret := C.cudaFree(devicePtr)
	err := (CudaError)(ret)
	return err
}

func FreeAsync(devicePtr unsafe.Pointer, stream Stream) CudaError {
	cStream := (C.cudaStream_t)(stream)
	ret := C.cudaFreeAsync(devicePtr, cStream)
	err := (CudaError)(ret)
	return err
}

func CopyToHost(hostDst, deviceSrc unsafe.Pointer, size uint) (unsafe.Pointer, CudaError) {
	cCount := (C.size_t)(size)
	ret := C.cudaMemcpy(hostDst, deviceSrc, cCount, uint32(CudaMemcpyDeviceToHost))
	err  := (CudaError)(ret)
	return hostDst, err
}

func CopyToHostAsync(hostDst, deviceSrc unsafe.Pointer, size uint, stream CudaStream) CudaError {
	cSize := (C.size_t)(size)
	cStream := (C.cudaStream_t)(stream)
	ret := C.cudaMemcpyAsync(hostDst, deviceSrc, cSize, uint32(CudaMemcpyDeviceToHost), cStream)
	err := (CudaError)(ret)
	return err
}

func CopyFromHost(deviceDst, hostSrc unsafe.Pointer, size uint) (unsafe.Pointer, CudaError) {
	cSize := (C.size_t)(size)
	ret := C.cudaMemcpy(deviceDst, hostSrc, cSize, uint32(CudaMemcpyHostToDevice))
	err := (CudaError)(ret)
	return deviceDst, err
}

func CopyFromHostAsync(deviceDst, hostSrc unsafe.Pointer, size uint, stream CudaStream) CudaError {
	cCount := (C.size_t)(size)
	cStream := (C.cudaStream_t)(stream)
	ret := C.cudaMemcpyAsync(deviceDst, hostSrc, cCount, uint32(CudaMemcpyHostToDevice), cStream)
	err := (CudaError)(ret)
	return err
}
