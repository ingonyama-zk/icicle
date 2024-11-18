package runtime

// #cgo CFLAGS: -I./include/
// #include "runtime.h"
import "C"
import (
	"runtime"
	"unsafe"
)

func LoadBackend(path string, isRecursive bool) EIcicleError {
	cPath := C.CString(path)
	cIsRecursive := C._Bool(isRecursive)
	cErr := C.icicle_load_backend(cPath, cIsRecursive)
	return EIcicleError(cErr)
}

func LoadBackendFromEnvOrDefault() EIcicleError {
	return EIcicleError(C.icicle_load_backend_from_env_or_default())
}

func WarmUpDevice() EIcicleError {
	mem, err := GetAvailableMemory()
	if err != Success {
		return EIcicleError(err)
	}

	allocatedMem, err := Malloc(mem.Free / 2)
	if err != Success {
		return EIcicleError(err)
	}

	Free(allocatedMem)
	return EIcicleError(Success)
}

type AvailableMemory struct {
	Total uint
	Free  uint
}

func GetAvailableMemory() (*AvailableMemory, EIcicleError) {
	memory := AvailableMemory{Total: 0, Free: 0}
	cTotal := (*C.size_t)(unsafe.Pointer(&memory.Total))
	cFree := (*C.size_t)(unsafe.Pointer(&memory.Free))
	cErr := C.icicle_get_available_memory(cTotal, cFree)
	err := EIcicleError(cErr)
	if err != Success {
		return nil, err
	}
	return &memory, err
}

func IsHostMemory(ptr unsafe.Pointer) bool {
	cErr := C.icicle_is_host_memory(ptr)
	return EIcicleError(cErr) == Success
}

func IsActiveDeviceMemory(ptr unsafe.Pointer) bool {
	cErr := C.icicle_is_active_device_memory(ptr)
	return EIcicleError(cErr) == Success
}

// RunOnDevice forces the provided function to run all device related calls within it
// on the same host thread and therefore the same device.
//
// NOTE: Goroutines launched within funcToRun are not bound to the
// same host thread as funcToRun and therefore not to the same device.
// If that is a requirement, RunOnDevice should be called for each with the
// same deviceId as the original call.
//
// As an example:
//
// cr.RunOnDevice(i, func(args ...any) {
// 		defer wg.Done()
// 		cfg := GetDefaultMSMConfig()
// 		stream, _ := cr.CreateStream()

// // This will always print "Inner goroutine device: 0"
// // go func ()  {
// // 	device, _ := runtime.GetActiveDevice()
// // 	fmt.Println("Inner goroutine device: ", device.Id)
// // }()
//
// // To force the above goroutine to same device as the wrapping function:
// // RunOnDevice(i, func(arg ...any) {
// // 	device, _ := runtime.GetActiveDevice()
// // 	fmt.Println("Inner goroutine device: ", device.Id)
// // })
// .
// .
// .
//
//	}, i)
func RunOnDevice(device *Device, funcToRun func(args ...any), args ...any) {
	go func(deviceToRunOn *Device) {
		defer runtime.UnlockOSThread()
		runtime.LockOSThread()
		originalDevice, _ := GetActiveDevice()
		SetDevice(deviceToRunOn)
		funcToRun(args...)
		SetDevice(originalDevice)
	}(device)
}
