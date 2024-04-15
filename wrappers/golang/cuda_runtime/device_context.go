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
	"fmt"
	"runtime"
	"unsafe"
)

type DeviceContext struct {
	/// Stream to use. Default value: 0.
	Stream *Stream // Assuming the type is provided by a CUDA binding crate

	/// Index of the currently used GPU. Default value: 0.
	deviceId uint

	/// Mempool to use. Default value: 0.
	Mempool MemPool // Assuming the type is provided by a CUDA binding crate
}

func (d DeviceContext) GetDeviceId() int {
	return int(d.deviceId)
}

func GetDefaultDeviceContext() (DeviceContext, CudaError) {
	device, err := GetDevice()
	if err != CudaSuccess {
		panic(fmt.Sprintf("Could not get current device due to %v", err))
	}
	var defaultStream Stream
	var defaultMempool MemPool

	return DeviceContext{
		&defaultStream,
		uint(device),
		defaultMempool,
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

func GetDevice() (int, CudaError) {
	var device int
	cDevice := (*C.int)(unsafe.Pointer(&device))
	err := C.cudaGetDevice(cDevice)
	return device, (CudaError)(err)
}

func GetDeviceFromPointer(ptr unsafe.Pointer) int {
	var cCudaPointerAttributes CudaPointerAttributes
	err := C.cudaPointerGetAttributes(&cCudaPointerAttributes, ptr)
	if (CudaError)(err) != CudaSuccess {
		panic("Could not get attributes of pointer")
	}
	return int(cCudaPointerAttributes.device)
}

// RunOnDevice forces the provided function to run all GPU related calls within it
// on the same host thread and therefore the same GPU device.
//
// NOTE: Goroutines launched within funcToRun are not bound to the
// same host thread as funcToRun and therefore not to the same GPU device.
// If that is a requirement, RunOnDevice should be called for each with the
// same deviceId as the original call.
//
// As an example:
//
//	   		cr.RunOnDevice(i, func(args ...any) {
//					 	defer wg.Done()
//					 	cfg := GetDefaultMSMConfig()
//					 	stream, _ := cr.CreateStream()
//					 	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
//					 		size := 1 << power
//
//					 		// This will always print "Inner goroutine device: 0"
//							// go func ()  {
//							// 	device, _ := cr.GetDevice()
//							// 	fmt.Println("Inner goroutine device: ", device)
//							// }()
//					 		// To force the above goroutine to same device as the wrapping function:
//							// RunOnDevice(i, func(arg ...any) {
//							// 	device, _ := cr.GetDevice()
//							// 	fmt.Println("Inner goroutine device: ", device)
//							// })
//
//					 		scalars := GenerateScalars(size)
//					 		points := GenerateAffinePoints(size)
//
//					 		var p Projective
//					 		var out core.DeviceSlice
//					 		_, e := out.MallocAsync(p.Size(), p.Size(), stream)
//					 		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
//					 		cfg.Ctx.Stream = &stream
//					 		cfg.IsAsync = true
//
//					 		e = Msm(scalars, points, &cfg, out)
//					 		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
//
//					 		outHost := make(core.HostSlice[Projective], 1)
//
//					 		cr.SynchronizeStream(&stream)
//					 		outHost.CopyFromDevice(&out)
//					 		out.Free()
//					 		// Check with gnark-crypto
//					 		assert.True(t, testAgainstGnarkCryptoMsm(scalars, points, outHost[0]))
//					 	}
//					}, i)
func RunOnDevice(deviceId int, funcToRun func(args ...any), args ...any) {
	go func(id int) {
		defer runtime.UnlockOSThread()
		runtime.LockOSThread()
		SetDevice(id)
		funcToRun(args...)
	}(deviceId)
}
