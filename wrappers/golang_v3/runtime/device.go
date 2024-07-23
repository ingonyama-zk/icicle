package runtime

import "C"
import "unsafe"

const MAX_TYPE_SIZE = 64

type Device struct {
	DeviceType [MAX_TYPE_SIZE]C.char
	Id         int
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
	return Device{DeviceType: cDeviceType, Id: id}
}

func (self *Device) GetDeviceType() string {
	n := 0
	for n < MAX_TYPE_SIZE && self.DeviceType[n] != 0 {
		n++
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(&self.DeviceType[0])), C.int(n))
}
