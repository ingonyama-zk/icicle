package runtime

// #cgo CFLAGS: -I./include/
// #include "runtime.h"
import "C"
import "unsafe"

func Malloc(size uint) (unsafe.Pointer, EIcicleError) {
	if size == 0 {
		return nil, AllocationFailed
	}

	var p C.void
	devicePtr := unsafe.Pointer(&p)
	cSize := (C.size_t)(size)

	ret := C.icicle_malloc(&devicePtr, cSize)
	err := EIcicleError(ret)

	return devicePtr, err
}

func MallocAsync(size uint, stream Stream) (unsafe.Pointer, EIcicleError) {
	if size == 0 {
		return nil, AllocationFailed
	}

	var p C.void
	devicePtr := unsafe.Pointer(&p)
	cSize := (C.size_t)(size)

	ret := C.icicle_malloc_async(&devicePtr, cSize, stream)
	err := EIcicleError(ret)

	return devicePtr, err
}

func Free(devicePtr unsafe.Pointer) EIcicleError {
	ret := C.icicle_free(devicePtr)
	err := EIcicleError(ret)
	return err
}

func FreeAsync(devicePtr unsafe.Pointer, stream Stream) EIcicleError {
	ret := C.icicle_free_async(devicePtr, stream)
	err := EIcicleError(ret)
	return err
}

func MemSet(devicePtr unsafe.Pointer, value int, size uint) EIcicleError {
	ret := C.icicle_memset(devicePtr, (C.int)(value), (C.size_t)(size))
	err := EIcicleError(ret)
	return err
}

func MemSetAsync(devicePtr unsafe.Pointer, value int, size uint, stream Stream) EIcicleError {
	ret := C.icicle_memset_async(devicePtr, (C.int)(value), (C.size_t)(size), stream)
	err := EIcicleError(ret)
	return err
}

func CopyFromDevice(hostDst, deviceSrc unsafe.Pointer, size uint) (unsafe.Pointer, EIcicleError) {
	cSize := (C.size_t)(size)
	ret := C.icicle_copy_to_host(hostDst, deviceSrc, cSize)
	err := (EIcicleError)(ret)
	return hostDst, err
}

func CopyFromDeviceAsync(hostDst, deviceSrc unsafe.Pointer, size uint, stream Stream) EIcicleError {
	cSize := (C.size_t)(size)
	ret := C.icicle_copy_to_host_async(hostDst, deviceSrc, cSize, stream)
	err := (EIcicleError)(ret)
	return err
}

func CopyToDevice(deviceDst, hostSrc unsafe.Pointer, size uint) (unsafe.Pointer, EIcicleError) {
	cSize := (C.size_t)(size)
	ret := C.icicle_copy_to_device(deviceDst, hostSrc, cSize)
	err := (EIcicleError)(ret)
	return deviceDst, err
}

func CopyToDeviceAsync(deviceDst, hostSrc unsafe.Pointer, size uint, stream Stream) EIcicleError {
	cSize := (C.size_t)(size)
	ret := C.icicle_copy_to_device_async(deviceDst, hostSrc, cSize, stream)
	err := (EIcicleError)(ret)
	return err
}
