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

type DeviceContext struct {
	/// Stream to use. Default value: 0.
	Stream *Stream // Assuming the type is provided by a CUDA binding crate

	/// Index of the currently used GPU. Default value: 0.
	DeviceId uint

	/// Mempool to use. Default value: 0.
	Mempool MemPool // Assuming the type is provided by a CUDA binding crate
}

func GetDefaultDeviceContext() (DeviceContext, CudaError) {
	defaultContext := GetDefaultDeviceContextForDevice(0)
	return defaultContext, CudaSuccess
}

func GetDefaultDeviceContextForDevice(deviceId int) DeviceContext {
	var defaultStream Stream
	var defaultMempool MemPool

	return DeviceContext{
		&defaultStream,
		uint(deviceId),
		defaultMempool,
	}
}

func SetDevice(device int) CudaError {
	cDevice := (C.int)(device)
	ret := C.cudaSetDevice(cDevice)
	err := (CudaError)(ret)
	return err
}

func GetDeviceCount() (int, CudaError) {
	var count int
	cCount := (*C.int)(unsafe.Pointer(&count))
	err := C.cudaGetDeviceCount(cCount)
	return count, (CudaError)(err)
}

func GetDevice() (int, CudaError) {
	var device int
	cDevice := (*C.int)(unsafe.Pointer(&device))
	err := C.cudaGetDevice(cDevice)
	return device, (CudaError)(err)
}
