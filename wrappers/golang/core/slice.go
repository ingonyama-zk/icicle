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

// CheckDevice is used to ensure that the DeviceSlice about to be used resides on the currently set device
func (d DeviceSlice) CheckDevice() {
	if currentDeviceId, err := cr.GetDevice(); err != cr.CudaSuccess || d.GetDeviceId() != currentDeviceId {
		panic("Attempt to use DeviceSlice on a different device")
	}
}

func(d *DeviceSlice) Range(start, end int, endInclusive bool) DeviceSlice {
	if end <= start {
		panic("Cannot have negative or zero size slices")
	}
	
	if end >= d.length {
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

func(d *DeviceSlice) RangeTo(end int, inclusive bool) DeviceSlice {
	if end <= 0 {
		panic("Cannot have negative or zero size slices")
	}

	if end >= d.length {
		panic("Cannot increase slice size from Range")
	}
	
	var newSlice DeviceSlice
	sizeOfElement := d.capacity/d.length
	newSlice.length = end
	if inclusive {
		newSlice.length += 1
	}
	newSlice.capacity = newSlice.length*sizeOfElement
	newSlice.inner = d.inner
	return newSlice
}

func(d *DeviceSlice) RangeFrom(start int) DeviceSlice {
	if start >= d.length {
		panic("Cannot have negative or zero size slices")
	}
	
	var newSlice DeviceSlice
	sizeOfElement := d.capacity/d.length

	newSlice.inner = unsafe.Pointer(uintptr(d.inner) + uintptr(start)*uintptr(sizeOfElement))
	newSlice.length = d.length - start
	newSlice.capacity = d.capacity - start*sizeOfElement

	return newSlice
}

// TODO: change signature to be Malloc(element, numElements)
// calc size internally
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

func (h HostSlice[T]) CopyToDevice(dst *DeviceSlice, shouldAllocate bool) *DeviceSlice {
	size := h.Len() * h.SizeOfElement()
	if shouldAllocate {
		dst.Malloc(size, h.SizeOfElement())
	}
	dst.CheckDevice()
	if size > dst.Cap() {
		panic("Number of bytes to copy is too large for destination")
	}

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
