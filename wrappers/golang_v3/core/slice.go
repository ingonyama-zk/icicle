package core

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
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

// func (d DeviceSlice) GetDeviceId() int {
// 	return runtime.GetDeviceFromPointer(d.inner)
// }

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

func (d *DeviceSlice) RangeTo(end int, inclusive bool) DeviceSlice {
	if end <= 0 {
		panic("Cannot have negative or zero size slices")
	}

	if end >= d.length {
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

// TODO: change signature to be Malloc(element, numElements)
// calc size internally
func (d *DeviceSlice) Malloc(size, sizeOfElement int) (DeviceSlice, runtime.EIcicleError) {
	dp, err := runtime.Malloc(uint(size))
	d.inner = dp
	d.capacity = size
	d.length = size / sizeOfElement
	return *d, err
}

func (d *DeviceSlice) MallocAsync(size, sizeOfElement int, stream runtime.Stream) (DeviceSlice, runtime.EIcicleError) {
	dp, err := runtime.MallocAsync(uint(size), stream)
	d.inner = dp
	d.capacity = size
	d.length = size / sizeOfElement
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

// Registers host memory as pinned, allowing the GPU to read data from the host quicker and save GPU memory space.
// Memory pinned using this function should be unpinned using [Unpin]
// func (h HostSlice[T]) Pin(flags runtime.RegisterPinnedFlags) runtime.EIcicleError {
// 	_, err := runtime.RegisterPinned(h.AsUnsafePointer(), h.SizeOfElement()*h.Len(), flags)
// 	return err
// }

// Unregisters host memory as pinned
// func (h HostSlice[T]) Unpin() runtime.EIcicleError {
// 	return runtime.FreeRegisteredPinned(h.AsUnsafePointer())
// }

// Allocates new host memory as pinned and copies the HostSlice data to the newly allocated area
// Memory pinned using this function should be unpinned using [FreePinned]
// func (h HostSlice[T]) AllocPinned(flags runtime.AllocPinnedFlags) (HostSlice[T], runtime.EIcicleError) {
// 	pinnedMemPointer, err := runtime.AllocPinned(h.SizeOfElement()*h.Len(), flags)
// 	if err != runtime.Success {
// 		return nil, err
// 	}
// 	pinnedMem := unsafe.Slice((*T)(pinnedMemPointer), h.Len())
// 	copy(pinnedMem, h)
// 	return pinnedMem, runtime.Success
// }

// Unpins host memory that was pinned using [AllocPinned]
// func (h HostSlice[T]) FreePinned() runtime.EIcicleError {
// 	return runtime.FreeAllocPinned(h.AsUnsafePointer())
// }

func (h HostSlice[T]) CopyToDevice(dst *DeviceSlice, shouldAllocate bool) *DeviceSlice {
	size := h.Len() * h.SizeOfElement()
	if shouldAllocate {
		dst.Malloc(size, h.SizeOfElement())
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
		dst.MallocAsync(size, h.SizeOfElement(), stream)
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
