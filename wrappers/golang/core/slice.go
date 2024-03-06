package core

import (
	"unsafe"

	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

type HostOrDeviceSlice interface {
	Len() int
	Cap() int
	IsEmpty() bool
	IsOnDevice() bool
}

type DevicePointer = unsafe.Pointer

type DeviceSlice struct {
	inner unsafe.Pointer
	// capacity is the number of bytes that have been allocated
	capacity int
	// length is the number of elements that have been written
	length int
}

func (d DeviceSlice) Len() int {
	return d.length
}

func (d DeviceSlice) Cap() int {
	return d.capacity
}

func (d DeviceSlice) IsEmpty() bool {
	return d.length == 0
}

func (d DeviceSlice) AsPointer() unsafe.Pointer {
	return d.inner
}

func (d DeviceSlice) IsOnDevice() bool {
	return true
}

func (d DeviceSlice) GetDeviceId() int {
	return cr.GetDeviceFromPointer(d.inner)
}

func (d DeviceSlice) CheckDevice() {
	if currentDeviceId, err := cr.GetDevice(); err != cr.CudaSuccess || d.GetDeviceId() != currentDeviceId {
		panic("Attempt to use DeviceSlice on a different device")
	}
}

func (d *DeviceSlice) Malloc(size, sizeOfElement int) (DeviceSlice, cr.CudaError) {
	dp, err := cr.Malloc(uint(size))
	d.inner = dp
	d.capacity = size
	d.length = size / sizeOfElement
	return *d, err
}

func (d *DeviceSlice) MallocAsync(size, sizeOfElement int, stream cr.CudaStream) (DeviceSlice, cr.CudaError) {
	dp, err := cr.MallocAsync(uint(size), stream)
	d.inner = dp
	d.capacity = size
	d.length = size / sizeOfElement
	return *d, err
}

func (d *DeviceSlice) Free() cr.CudaError {
	d.CheckDevice()
	err := cr.Free(d.inner)
	if err == cr.CudaSuccess {
		d.length, d.capacity = 0, 0
		d.inner = nil
	}
	return err
}

func (d *DeviceSlice) FreeAsync(stream cr.Stream) cr.CudaError {
	d.CheckDevice()
	err := cr.FreeAsync(d.inner, stream)
	if err == cr.CudaSuccess {
		d.length, d.capacity = 0, 0
		d.inner = nil
	}
	return err
}

type HostSliceInterface interface {
	Size() int
}

type HostSlice[T HostSliceInterface] []T

func HostSliceFromElements[T HostSliceInterface](elements []T) HostSlice[T] {
	slice := make(HostSlice[T], len(elements))
	copy(slice, elements)

	return slice
}

func HostSliceWithValue[T HostSliceInterface](underlyingValue T, size int) HostSlice[T] {
	slice := make(HostSlice[T], size)
	for i := range slice {
		slice[i] = underlyingValue
	}

	return slice
}

func (h HostSlice[T]) Len() int {
	return len(h)
}

func (h HostSlice[T]) Cap() int {
	return cap(h)
}

func (h HostSlice[T]) IsEmpty() bool {
	return len(h) == 0
}

func (h HostSlice[T]) IsOnDevice() bool {
	return false
}

func (h HostSlice[T]) SizeOfElement() int {
	return h[0].Size()
}

func (h HostSlice[T]) CopyToDevice(dst *DeviceSlice, shouldAllocate bool) *DeviceSlice {
	size := h.Len() * h.SizeOfElement()
	if shouldAllocate {
		dst.Malloc(size, h.SizeOfElement())
	}
	dst.CheckDevice()
	if size > dst.Cap() {
		panic("Number of bytes to copy is too large for destination")
	}

	// hostSrc := unsafe.Pointer(h.AsPointer())
	hostSrc := unsafe.Pointer(&h[0])
	cr.CopyToDevice(dst.inner, hostSrc, uint(size))
	dst.length = h.Len()
	return dst
}

func (h HostSlice[T]) CopyToDeviceAsync(dst *DeviceSlice, stream cr.CudaStream, shouldAllocate bool) *DeviceSlice {
	size := h.Len() * h.SizeOfElement()
	if shouldAllocate {
		dst.MallocAsync(size, h.SizeOfElement(), stream)
	}
	dst.CheckDevice()
	if size > dst.Cap() {
		panic("Number of bytes to copy is too large for destination")
	}

	hostSrc := unsafe.Pointer(&h[0])
	cr.CopyToDeviceAsync(dst.inner, hostSrc, uint(size), stream)
	dst.length = h.Len()
	return dst
}

func (h HostSlice[T]) CopyFromDevice(src *DeviceSlice) {
	src.CheckDevice()
	if h.Len() != src.Len() {
		panic("destination and source slices have different lengths")
	}
	bytesSize := src.Len() * h.SizeOfElement()
	cr.CopyFromDevice(unsafe.Pointer(&h[0]), src.inner, uint(bytesSize))
}

func (h HostSlice[T]) CopyFromDeviceAsync(src *DeviceSlice, stream cr.Stream) {
	src.CheckDevice()
	if h.Len() != src.Len() {
		panic("destination and source slices have different lengths")
	}
	bytesSize := src.Len() * h.SizeOfElement()
	cr.CopyFromDeviceAsync(unsafe.Pointer(&h[0]), src.inner, uint(bytesSize), stream)
}
