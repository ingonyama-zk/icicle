package runtime

// #cgo CFLAGS: -I./include/
// #include "runtime.h"
import "C"
import (
	"strings"
	"unsafe"
)

const MAX_TYPE_SIZE = 64

type Device struct {
	DeviceType [MAX_TYPE_SIZE]C.char
	Id         int32
}

type DeviceProperties struct {
	UsingHostMemory      bool
	NumMemoryRegions     int32
	SupportsPinnedMemory bool
}

func CreateDevice(deviceType string, id int) Device {
	var cDeviceType [MAX_TYPE_SIZE]C.char
	for i, v := range deviceType {
		if i >= MAX_TYPE_SIZE {
			break
		}
		cDeviceType[i] = C.char(v)
	}
	// Ensure the last character is null if the source string is too long
	if len(deviceType) >= MAX_TYPE_SIZE {
		cDeviceType[MAX_TYPE_SIZE-1] = C.char(0)
	}
	return Device{DeviceType: cDeviceType, Id: int32(id)}
}

func (self *Device) GetDeviceType() string {
	n := 0
	for n < MAX_TYPE_SIZE && self.DeviceType[n] != 0 {
		n++
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(&self.DeviceType[0])), C.int(n))
}

func SetDevice(device *Device) EIcicleError {
	cDevice := (*C.Device)(unsafe.Pointer(device))
	cErr := C.icicle_set_device(cDevice)
	return EIcicleError(cErr)
}

func SetDefaultDevice(device *Device) EIcicleError {
	cDevice := (*C.Device)(unsafe.Pointer(device))
	cErr := C.icicle_set_default_device(cDevice)
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
	cErr := C.icicle_is_device_available(cDevice)
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
	var properties DeviceProperties
	cProperties := (*C.DeviceProperties)(unsafe.Pointer(&properties))
	cErr := C.icicle_get_device_properties(cProperties)
	err := EIcicleError(cErr)
	if err != Success {
		return nil, err
	}
	return &properties, err
}
