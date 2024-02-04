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
	// TODO: use cuda_bindings.CudaMemPool as type
	Mempool uint // Assuming the type is provided by a CUDA binding crate
}

func GetDefaultDeviceContext() (DeviceContext, CudaError) {
	var defaultStream Stream

	return DeviceContext {
			&defaultStream,
			0,
			0,
	}, CudaSuccess
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
