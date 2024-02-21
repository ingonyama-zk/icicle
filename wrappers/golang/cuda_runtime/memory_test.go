package cuda_runtime

import (
	"testing"

	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestMalloc(t *testing.T) {
	_, err := Malloc(20)
	assert.Equal(t, CudaSuccess, err, "Unable to allocate device memory due to %d", err)
}

func TestMallocAsync(t *testing.T) {
	stream, _ := CreateStream()
	_, err := MallocAsync(20, stream)
	assert.Equal(t, CudaSuccess, err, "Unable to allocate device memory due to %d", err)
}

func TestFree(t *testing.T) {
	mem, err := Malloc(20)
	assert.Equal(t, CudaSuccess, err, "Unable to allocate device memory due to %d", err)

	err = Free(mem)
	assert.Equal(t, CudaSuccess, err, "Unable to free device memory due to %v", err)
}

func TestCopyFromToHost(t *testing.T) {
	someInts := make([]int32, 1)
	someInts[0] = 34
	numBytes := uint(8)
	deviceMem, _ := Malloc(numBytes)
	deviceMem, err := CopyToDevice(deviceMem, unsafe.Pointer(&someInts[0]), numBytes)
	assert.Equal(t, CudaSuccess, err, "Couldn't copy to device due to %v", err)

	someInts2 := make([]int32, 1)
	_, err = CopyFromDevice(unsafe.Pointer(&someInts2[0]), deviceMem, numBytes)
	assert.Equal(t, CudaSuccess, err, "Couldn't copy to device due to %v", err)
	assert.Equal(t, someInts, someInts2, "Elements of host slices do not match. Copying from/to host failed")
}
