package core

import (
	"local/hello/icicle/wrappers/golang/cuda_runtime"
	"unsafe"
)

type DeviceSlice struct {
	inner unsafe.Pointer
	capacity int
	length int
}

func (d* DeviceSlice) Len() int {
	return d.length
}

func (d* DeviceSlice) Cap() int {
	return d.capacity
}

func (d* DeviceSlice) IsEmpty() bool {
	return d.length == 0
}

func (d* DeviceSlice) AsSlice() []any {
	panic("Use CopyToHost or CopyToHostAsync to move device data to a slice")
}

func (d* DeviceSlice) AsPointer() any {
	return d.inner
}

func (d *DeviceSlice) IsOnDevice() bool {
	return true
}

func (d* DeviceSlice) Malloc(size uint) (DeviceSlice, cuda_runtime.CudaError) {
	dp, err := cuda_runtime.Malloc(size)
	return DeviceSlice{inner: dp, capacity: int(size), length: 0}, err
}

func (d* DeviceSlice) MallocAsync(size uint, stream cuda_runtime.CudaStream) (DeviceSlice, cuda_runtime.CudaError) {
	dp, err := cuda_runtime.Malloc(size)
	return DeviceSlice{inner: dp, capacity: int(size), length: 0}, err
}

func (d* DeviceSlice) Free() cuda_runtime.CudaError{
	err := cuda_runtime.Free(d.inner)
	if err == cuda_runtime.CudaSuccess {
		d.length, d.capacity = 0, 0
	}
	return err
}

func (d* DeviceSlice) CopyToHost(dst *HostSlice, size uint) *HostSlice {
	numElems := size / uint(dst.SizeOfElement())
	if numElems > uint(dst.Cap()) {
		panic("Number of elements to copy is too large for destination")
	}

	cuda_runtime.CopyToHost(unsafe.Pointer(dst.AsPointer()), d.inner, size)
	return dst
}

func (d* DeviceSlice) CopyToHostAsync(dst *HostSlice, size uint, stream cuda_runtime.CudaStream) {
	numElems := size / uint(dst.SizeOfElement())
	if numElems > uint(dst.Cap()) {
		panic("Number of elements to copy is too large for destination")
	}

	cuda_runtime.CopyToHostAsync(unsafe.Pointer(dst.AsPointer()), d.inner, size, stream)
}


type HostSlice []Field

func NewHostSlice(elements []Field) HostSlice {
	slice := make(HostSlice, len(elements))
	copy(slice, elements)

	return slice
}

func (h *HostSlice) Len() int {
	return len(*h)
}

func (h *HostSlice) Cap() int {
	return cap(*h)
}

func (h *HostSlice) IsEmpty() bool {
	return len(*h) == 0
}

func (h *HostSlice) AsSlice() []Field {
		return *h
}

func (h *HostSlice) AsPointer() *uint32 {
	return &((*h)[0].Limbs[0])
}

func (h *HostSlice) IsOnDevice() bool {
	return false
}

func (h *HostSlice) SizeOfElement() int {
	return (*h)[0].Size()
}

func (h *HostSlice) CopyFromHost(dst *DeviceSlice, size uint) *DeviceSlice {
	numElems := size / uint(h.SizeOfElement())
	if numElems > uint(dst.Cap()) {
		panic("Number of elements to copy is too large for destination")
	}

	hostSrc := unsafe.Pointer(h.AsPointer())
	cuda_runtime.CopyFromHost(dst.inner, hostSrc, size)
	dst.length = int(numElems)
	return dst
}

func (h* HostSlice) CopyFromHostAsync(dst *DeviceSlice, size uint, stream cuda_runtime.CudaStream) *DeviceSlice {
	numElems := size / uint(h.SizeOfElement())
	if numElems > uint(dst.Cap()) {
		panic("Number of elements to copy is too large for destination")
	}

	hostSrc := unsafe.Pointer(h.AsPointer())
	cuda_runtime.CopyFromHostAsync(dst.inner, hostSrc, size, stream)
	dst.length = int(numElems)
	return dst
}