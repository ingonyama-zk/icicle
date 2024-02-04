package cuda_runtime

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetDefaultContext(t *testing.T) {
	defaultCtx, err := GetDefaultDeviceContext()

	assert.Equal(t, CudaSuccess, err, "Couldn't get default device context due to %v", err)
	assert.Nil(t, *defaultCtx.Stream)
}

func TestSetDevice(t *testing.T) {
	err := SetDevice(0)
	assert.Equal(t, CudaSuccess, err, "Couldn't set device due to %v", err)
	
	count, _ := GetDeviceCount()
	err = SetDevice(count)
	assert.Equal(t, CudaErrorInvalidDevice, err, "Couldn't set device due to %v", err)
}

func TestGetDeviceCount(t *testing.T) {
	count, err := GetDeviceCount()
	assert.Equal(t, CudaSuccess, err, "Could not get device count due to %v", err)
	assert.GreaterOrEqual(t, count, 1, "Number of devices is 0, expected at least 1")
}
