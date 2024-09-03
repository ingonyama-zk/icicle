package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/stretchr/testify/assert"
)

func TestGetDeviceType(t *testing.T) {
	expectedDeviceName := "test"
	config := runtime.CreateDevice(expectedDeviceName, 0)
	assert.Equal(t, expectedDeviceName, config.GetDeviceType())

	expectedDeviceNameLong := "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
	configLargeName := runtime.CreateDevice(expectedDeviceNameLong, 1)
	assert.NotEqual(t, expectedDeviceNameLong, configLargeName.GetDeviceType())
}

func TestIsDeviceAvailable(t *testing.T) {
	runtime.LoadBackendFromEnvOrDefault()
	dev := runtime.CreateDevice("CUDA", 0)
	err := runtime.SetDevice(&dev)
	res, err := runtime.GetDeviceCount()
	assert.Equal(t, runtime.Success, err)
	assert.Equal(t, res, 2)

	err = runtime.LoadBackendFromEnvOrDefault()
	assert.Equal(t, runtime.Success, err)
	devCuda := runtime.CreateDevice("CUDA", 0)
	assert.True(t, runtime.IsDeviceAvailable(&devCuda))
	devCpu := runtime.CreateDevice("CPU", 0)
	assert.True(t, runtime.IsDeviceAvailable(&devCpu))
	devInvalid := runtime.CreateDevice("invalid", 0)
	assert.False(t, runtime.IsDeviceAvailable(&devInvalid))
}

func TestRegisteredDevices(t *testing.T) {
	err := runtime.LoadBackendFromEnvOrDefault()
	assert.Equal(t, runtime.Success, err)
	devices, err := runtime.GetRegisteredDevices()
	assert.Equal(t, []string{"CUDA", "CPU"}, devices)
}

func TestDeviceProperties(t *testing.T) {
	_, err := runtime.GetDeviceProperties()
	assert.Equal(t, runtime.Success, err)
}

func TestActiveDevice(t *testing.T) {
	runtime.SetDevice(&DEVICE)
	activeDevice, err := runtime.GetActiveDevice()
	assert.Equal(t, runtime.Success, err)
	assert.Equal(t, DEVICE, *activeDevice)
	memory1, err := runtime.GetAvailableMemory()
	if err == runtime.ApiNotImplemented {
		t.Skipf("GetAvailableMemory() function is not implemented on %s device", DEVICE.GetDeviceType())
	}
	assert.Equal(t, runtime.Success, err)
	assert.Greater(t, memory1.Total, uint(0))
	assert.Greater(t, memory1.Free, uint(0))
}
