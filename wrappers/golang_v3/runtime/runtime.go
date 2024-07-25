package runtime

// #cgo CFLAGS: -I./include/
// #include "runtime.h"
import "C"
import (
	"os"
	"runtime"
	"strings"
	"unsafe"
)

func LoadBackend(path string, isRecursive bool) EIcicleError {
	cPath := C.CString(path)
	cIsRecursive := C._Bool(isRecursive)
	cErr := C.icicle_load_backend(cPath, cIsRecursive)
	return EIcicleError(cErr)
}

func LoadBackendFromEnv() EIcicleError {
	path := os.Getenv("DEFAULT_BACKEND_INSTALL_DIR")
	return LoadBackend(path, true)
}

func SetDevice(device *Device) EIcicleError {
	cDevice := (*C.Device)(unsafe.Pointer(device))
	cErr := C.icicle_set_device(cDevice)
	return EIcicleError(cErr)
}

func GetActiveDevice() (*Device, EIcicleError) {
	device := CreateDevice("invalid", -1)
	cDevice := (*C.Device)(unsafe.Pointer(&device))
	cErr := C.icicle_get_active_device(cDevice)
	err := EIcicleError(cErr)
	if err != Success {
		return nil, err
	}
	return &device, err
}

func IsDeviceAvailable(device *Device) bool {
	cDevice := (*C.Device)(unsafe.Pointer(device))
	cErr := C.icicle_is_device_avialable(cDevice)
	return EIcicleError(cErr) == Success
}

func GetDeviceCount() (int, EIcicleError) {
	res := 0
	cRes := (*C.int)(unsafe.Pointer(&res))
	cErr := C.icicle_get_device_count(cRes)
	return res, EIcicleError(cErr)
}

func GetRegisteredDevices() ([]string, EIcicleError) {
	const BUFFER_SIZE = 256
	var buffer [BUFFER_SIZE]C.char
	cErr := C.icicle_get_registered_devices((*C.char)(unsafe.Pointer(&buffer[0])), BUFFER_SIZE)
	err := EIcicleError(cErr)
	if err != Success {
		return nil, err
	}
	n := 0
	for n < BUFFER_SIZE && buffer[n] != 0 {
		n++
	}
	res := C.GoStringN((*C.char)(unsafe.Pointer(&buffer[0])), C.int(n))
	return strings.Split(res, ","), err
}

func DeviceSynchronize() EIcicleError {
	cErr := C.icicle_device_synchronize()
	return EIcicleError(cErr)
}

func GetDeviceProperties() (*DeviceProperties, EIcicleError) {
	properties := DeviceProperties{
		UsingHostMemory:      false,
		NumMemoryRegions:     0,
		SupportsPinnedMemory: false,
	}
	cProperties := (*C.DeviceProperties)(unsafe.Pointer(&properties))
	cErr := C.icicle_get_device_properties(cProperties)
	err := EIcicleError(cErr)
	if err != Success {
		return nil, err
	}
	return &properties, err
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
// cr.RunOnDevice(i, func(args ...any) {
// 		defer wg.Done()
// 		cfg := GetDefaultMSMConfig()
// 		stream, _ := cr.CreateStream()
// 		for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
// 			size := 1 << power

// 			// This will always print "Inner goroutine device: 0"
// 			// go func ()  {
// 			// 	device, _ := cr.GetDevice()
// 			// 	fmt.Println("Inner goroutine device: ", device)
// 			// }()
// 			// To force the above goroutine to same device as the wrapping function:
// 			// RunOnDevice(i, func(arg ...any) {
// 			// 	device, _ := cr.GetDevice()
// 			// 	fmt.Println("Inner goroutine device: ", device)
// 			// })

// 			scalars := GenerateScalars(size)
// 			points := GenerateAffinePoints(size)

// 			var p Projective
// 			var out core.DeviceSlice
// 			_, e := out.MallocAsync(p.Size(), p.Size(), stream)
// 			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
// 			cfg.Ctx.Stream = &stream
// 			cfg.IsAsync = true

// 			e = Msm(scalars, points, &cfg, out)
// 			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")

// 			outHost := make(core.HostSlice[Projective], 1)

//			cr.SynchronizeStream(&stream)
//			outHost.CopyFromDevice(&out)
//			out.Free()
//			// Check with gnark-crypto
//			assert.True(t, testAgainstGnarkCryptoMsm(scalars, points, outHost[0]))
//		}
//	}, i)
func RunOnDevice(device *Device, funcToRun func(args ...any), args ...any) {
	go func(id *Device) {
		defer runtime.UnlockOSThread()
		runtime.LockOSThread()
		originalDevice, _ := GetActiveDevice()
		SetDevice(id)
		funcToRun(args...)
		SetDevice(originalDevice)
	}(device)
}
