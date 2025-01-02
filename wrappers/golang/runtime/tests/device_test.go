package tests

import (
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"

	icicle_runtime "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/stretchr/testify/assert"
)

func TestGetDeviceType(t *testing.T) {
	expectedDeviceName := "test"
	config := icicle_runtime.CreateDevice(expectedDeviceName, 0)
	assert.Equal(t, expectedDeviceName, config.GetDeviceType())

	expectedDeviceNameLong := "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
	configLargeName := icicle_runtime.CreateDevice(expectedDeviceNameLong, 1)
	assert.NotEqual(t, expectedDeviceNameLong, configLargeName.GetDeviceType())
}

func TestIsDeviceAvailable(t *testing.T) {
	dev := icicle_runtime.CreateDevice("CUDA", 0)
	_ = icicle_runtime.SetDevice(&dev)
	res, err := icicle_runtime.GetDeviceCount()

	smiCommand := exec.Command("nvidia-smi", "-L")
	smiCommandStdout, _ := smiCommand.StdoutPipe()
	wcCommand := exec.Command("wc", "-l")
	wcCommand.Stdin = smiCommandStdout

	smiCommand.Start()

	expectedNumDevicesRaw, wcErr := wcCommand.Output()
	smiCommand.Wait()

	expectedNumDevicesAsString := strings.TrimRight(string(expectedNumDevicesRaw), " \n\r\t")
	expectedNumDevices, _ := strconv.Atoi(expectedNumDevicesAsString)
	if wcErr != nil {
		t.Skip("Failed to get number of devices:", wcErr)
	}

	assert.Equal(t, icicle_runtime.Success, err)
	assert.Equal(t, expectedNumDevices, res)

	assert.Equal(t, icicle_runtime.Success, err)
	devCuda := icicle_runtime.CreateDevice("CUDA", 0)
	assert.True(t, icicle_runtime.IsDeviceAvailable(&devCuda))
	devCpu := icicle_runtime.CreateDevice("CPU", 0)
	assert.True(t, icicle_runtime.IsDeviceAvailable(&devCpu))
	devInvalid := icicle_runtime.CreateDevice("invalid", 0)
	assert.False(t, icicle_runtime.IsDeviceAvailable(&devInvalid))
}

func TestSetDefaultDevice(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	tidOuter := syscall.Gettid()

	gpuDevice := icicle_runtime.CreateDevice("CUDA", 0)
	icicle_runtime.SetDefaultDevice(&gpuDevice)

	activeDevice, err := icicle_runtime.GetActiveDevice()
	assert.Equal(t, icicle_runtime.Success, err)
	assert.Equal(t, gpuDevice, *activeDevice)

	done := make(chan struct{}, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		// Ensure we are operating on an OS thread other than the original one
		tidInner := syscall.Gettid()
		for tidInner == tidOuter {
			fmt.Println("Locked thread is the same as original, getting new locked thread")
			runtime.UnlockOSThread()
			runtime.LockOSThread()
			tidInner = syscall.Gettid()
		}

		activeDevice, err := icicle_runtime.GetActiveDevice()
		assert.Equal(t, icicle_runtime.Success, err)
		assert.Equal(t, gpuDevice, *activeDevice)

		close(done)
	}()

	<-done

	cpuDevice := icicle_runtime.CreateDevice("CPU", 0)
	icicle_runtime.SetDefaultDevice(&cpuDevice)
}

func TestRegisteredDevices(t *testing.T) {
	devices, _ := icicle_runtime.GetRegisteredDevices()
	assert.Equal(t, []string{"CUDA", "CPU"}, devices)
}

func TestDeviceProperties(t *testing.T) {
	_, err := icicle_runtime.GetDeviceProperties()
	assert.Equal(t, icicle_runtime.Success, err)
}

func TestActiveDevice(t *testing.T) {
	devCpu := icicle_runtime.CreateDevice("CUDA", 0)
	icicle_runtime.SetDevice(&devCpu)
	activeDevice, err := icicle_runtime.GetActiveDevice()
	assert.Equal(t, icicle_runtime.Success, err)
	assert.Equal(t, devCpu, *activeDevice)
	memory1, err := icicle_runtime.GetAvailableMemory()
	if err == icicle_runtime.ApiNotImplemented {
		t.Skipf("GetAvailableMemory() function is not implemented on %s device", devCpu.GetDeviceType())
	}
	assert.Equal(t, icicle_runtime.Success, err)
	assert.Greater(t, memory1.Total, uint(0))
	assert.Greater(t, memory1.Free, uint(0))
}
