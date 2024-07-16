package runtime

// #cgo CFLAGS: -I./include/
// #include "runtime.h"
import "C"
import (
	"os"
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
	path = "/home/administrator/users/Timur/Projects/icicle/icicle_v3/build"
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
