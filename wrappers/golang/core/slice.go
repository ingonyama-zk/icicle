package core

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

type HostOrDeviceSlice interface {
	Len() int
	Cap() int
	IsEmpty() bool
	IsOnDevice() bool
	AsUnsafePointer() unsafe.Pointer
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

func (d DeviceSlice) AsUnsafePointer() unsafe.Pointer {
	return d.inner
}

func (d DeviceSlice) IsOnDevice() bool {
	return true
}

// CheckDevice is used to ensure that the DeviceSlice about to be used resides on the currently set device
func (d DeviceSlice) CheckDevice() {
	if !runtime.IsActiveDeviceMemory(d.AsUnsafePointer()) {
		panic("Attempt to use DeviceSlice on a different device")
	}
}

func (d *DeviceSlice) Range(start, end int, endInclusive bool) DeviceSlice {
	if end <= start {
		panic("Cannot have negative or zero size slices")
	}

	if (endInclusive && end >= d.length) || (!endInclusive && end > d.length) {
		panic("Cannot increase slice size from Range")
	}

	var newSlice DeviceSlice
	switch {
	case start < 0:
		panic("Negative value for start is not supported")
	case start == 0:
		newSlice = d.RangeTo(end, endInclusive)
	case start > 0:
		tempSlice := d.RangeFrom(start)
		newSlice = tempSlice.RangeTo(end-start, endInclusive)
	}
	return newSlice
}

func (d *DeviceSlice) RangeTo(end int, inclusive bool) DeviceSlice {
	if end <= 0 {
		panic("Cannot have negative or zero size slices")
	}

	if (inclusive && end >= d.length) || (!inclusive && end > d.length) {
		panic("Cannot increase slice size from Range")
	}

	var newSlice DeviceSlice
	sizeOfElement := d.capacity / d.length
	newSlice.length = end
	if inclusive {
		newSlice.length += 1
	}
	newSlice.capacity = newSlice.length * sizeOfElement
	newSlice.inner = d.inner
	return newSlice
}

func (d *DeviceSlice) RangeFrom(start int) DeviceSlice {
	if start >= d.length {
		panic("Cannot have negative or zero size slices")
	}

	if start < 0 {
		panic("Negative value for start is not supported")
	}

	var newSlice DeviceSlice
	sizeOfElement := d.capacity / d.length

	newSlice.inner = unsafe.Pointer(uintptr(d.inner) + uintptr(start)*uintptr(sizeOfElement))
	newSlice.length = d.length - start
	newSlice.capacity = d.capacity - start*sizeOfElement

	return newSlice
}

func (d *DeviceSlice) Malloc(elementSize, numElements int) (DeviceSlice, runtime.EIcicleError) {
	dp, err := runtime.Malloc(uint(elementSize * numElements))
	d.inner = dp
	d.capacity = elementSize * numElements
	d.length = numElements
	return *d, err
}

func (d *DeviceSlice) MallocAsync(elementSize, numElements int, stream runtime.Stream) (DeviceSlice, runtime.EIcicleError) {
	dp, err := runtime.MallocAsync(uint(elementSize*numElements), stream)
	d.inner = dp
	d.capacity = elementSize * numElements
	d.length = numElements
	return *d, err
}

func (d *DeviceSlice) Free() runtime.EIcicleError {
	d.CheckDevice()
	err := runtime.Free(d.inner)
	if err == runtime.Success {
		d.length, d.capacity = 0, 0
		d.inner = nil
	}
	return err
}

func (d *DeviceSlice) FreeAsync(stream runtime.Stream) runtime.EIcicleError {
	d.CheckDevice()
	err := runtime.FreeAsync(d.inner, stream)
	if err == runtime.Success {
		d.length, d.capacity = 0, 0
		d.inner = nil
	}
	return err
}

type HostSlice[T any] []T

func HostSliceFromElements[T any](elements []T) HostSlice[T] {
	return elements
}

func HostSliceWithValue[T any](underlyingValue T, size int) HostSlice[T] {
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
	return int(unsafe.Sizeof(h[0]))
}

func (h HostSlice[T]) AsPointer() *T {
	return &h[0]
}

func (h HostSlice[T]) AsUnsafePointer() unsafe.Pointer {
	return unsafe.Pointer(&h[0])
}

func (h HostSlice[T]) CopyToDevice(dst *DeviceSlice, shouldAllocate bool) *DeviceSlice {
	size := h.Len() * h.SizeOfElement()
	if shouldAllocate {
		dst.Malloc(h.SizeOfElement(), h.Len())
	}
	dst.CheckDevice()
	if size > dst.Cap() {
		panic("Number of bytes to copy is too large for destination")
	}

	runtime.CopyToDevice(dst.inner, h.AsUnsafePointer(), uint(size))
	dst.length = h.Len()
	return dst
}

func (h HostSlice[T]) CopyToDeviceAsync(dst *DeviceSlice, stream runtime.Stream, shouldAllocate bool) *DeviceSlice {
	size := h.Len() * h.SizeOfElement()
	if shouldAllocate {
		dst.MallocAsync(h.SizeOfElement(), h.Len(), stream)
	}
	dst.CheckDevice()
	if size > dst.Cap() {
		panic("Number of bytes to copy is too large for destination")
	}

	runtime.CopyToDeviceAsync(dst.inner, h.AsUnsafePointer(), uint(size), stream)
	dst.length = h.Len()
	return dst
}

func (h HostSlice[T]) CopyFromDevice(src *DeviceSlice) {
	src.CheckDevice()
	if h.Len() != src.Len() {
		panic("destination and source slices have different lengths")
	}
	bytesSize := src.Len() * h.SizeOfElement()
	runtime.CopyFromDevice(h.AsUnsafePointer(), src.inner, uint(bytesSize))
}

func (h HostSlice[T]) CopyFromDeviceAsync(src *DeviceSlice, stream runtime.Stream) {
	src.CheckDevice()
	if h.Len() != src.Len() {
		panic("destination and source slices have different lengths")
	}
	bytesSize := src.Len() * h.SizeOfElement()
	runtime.CopyFromDeviceAsync(h.AsUnsafePointer(), src.inner, uint(bytesSize), stream)
}
