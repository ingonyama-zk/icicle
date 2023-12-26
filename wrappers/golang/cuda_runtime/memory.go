package cuda_runtime

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
	dp := unsafe.Pointer(&p)
	err := cudaMalloc(dp, uint64(size))

	return dp, err
}

func MallocAsync(size uint, stream CudaStream) (unsafe.Pointer, CudaError) {
	if size == 0 {
		return nil, CudaErrorMemoryAllocation
	}

	var p C.void
	dp := unsafe.Pointer(&p)
	err := cudaMallocAsync(dp, uint64(size), stream)

	return dp, err
}

func Free(d unsafe.Pointer) CudaError {
	return cudaFree(d)
}

func CopyToHost[T any](hostDst []T, deviceSrc unsafe.Pointer, size uint) (unsafe.Pointer, CudaError) {
	cHostDst := unsafe.Pointer(&hostDst[0])
	err := cudaMemcpy(cHostDst, deviceSrc, uint64(size), CudaMemcpyDeviceToHost)
	return cHostDst, err
}

func CopyToHostAsync(hostDst, deviceSrc unsafe.Pointer, size uint, stream CudaStream) unsafe.Pointer {
	cudaMemcpyAsync(hostDst, deviceSrc, int(size), CudaMemcpyDeviceToHost, stream)
	return hostDst
}

func CopyFromHost(deviceDst, hostSrc unsafe.Pointer, size uint) (unsafe.Pointer, CudaError) {
	err := cudaMemcpy(deviceDst, hostSrc, uint64(size), CudaMemcpyHostToDevice)
	return deviceDst, err
}

func CopyFromHostAsync(deviceDst, hostSrc unsafe.Pointer, size uint, stream CudaStream) unsafe.Pointer {
	cudaMemcpyAsync(deviceDst, hostSrc, int(size), CudaMemcpyHostToDevice, stream)
	return deviceDst
}
