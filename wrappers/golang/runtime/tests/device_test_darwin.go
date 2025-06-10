package tests

/*
#include <pthread.h>
#include <stdio.h> // For printing pthread_t if it's an integer type

// On some systems, pthread_t might be a struct or opaque type.
// To print it as an integer, you might need to cast it.
// This is a common way to get a numerical representation for logging/debugging.
unsigned long get_pthread_id() {
    return (unsigned long)pthread_self();
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	icicle_runtime "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/stretchr/testify/suite"
)

func testGetDeviceType(suite *suite.Suite) {
	expectedDeviceName := "test"
	config := icicle_runtime.CreateDevice(expectedDeviceName, 0)
	suite.Equal(expectedDeviceName, config.GetDeviceType())

	expectedDeviceNameLong := "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
	configLargeName := icicle_runtime.CreateDevice(expectedDeviceNameLong, 1)
	suite.NotEqual(expectedDeviceNameLong, configLargeName.GetDeviceType())
}

func testIsDeviceAvailable(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()
	res, err := icicle_runtime.GetDeviceCount()

	suite.Equal(icicle_runtime.Success, err)
	suite.Equal(1, res) // METAL only has one device

	suite.True(icicle_runtime.IsDeviceAvailable(&test_helpers.MAIN_DEVICE))
	suite.True(icicle_runtime.IsDeviceAvailable(&test_helpers.REFERENCE_DEVICE))
	devInvalid := icicle_runtime.CreateDevice("invalid", 0)
	suite.False(icicle_runtime.IsDeviceAvailable(&devInvalid))
}

func testSetDefaultDevice(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()

	activeDevice, err := icicle_runtime.GetActiveDevice()
	suite.Equal(icicle_runtime.Success, err)
	suite.Equal(test_helpers.MAIN_DEVICE, *activeDevice)

	outerThreadID := C.get_pthread_id()
	done := make(chan struct{}, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		// Ensure we are operating on an OS thread other than the original one
		tidInner := C.get_pthread_id()
		for tidInner == outerThreadID {
			fmt.Println("Locked thread is the same as original, getting new locked thread")
			runtime.UnlockOSThread()
			runtime.LockOSThread()
			tidInner = C.get_pthread_id()
		}

		activeDevice, err := icicle_runtime.GetActiveDevice()
		suite.Equal(icicle_runtime.Success, err)
		suite.Equal(test_helpers.MAIN_DEVICE, *activeDevice)

		close(done)
	}()

	<-done

	icicle_runtime.SetDefaultDevice(&test_helpers.REFERENCE_DEVICE)
}

func testRegisteredDevices(suite *suite.Suite) {
	devices, _ := icicle_runtime.GetRegisteredDevices()
	suite.Equal([]string{test_helpers.MAIN_DEVICE.GetDeviceType(), "CPU"}, devices)
}

func testDeviceProperties(suite *suite.Suite) {
	_, err := icicle_runtime.GetDeviceProperties()
	suite.Equal(icicle_runtime.Success, err)
}

func testActiveDevice(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()
	activeDevice, err := icicle_runtime.GetActiveDevice()
	suite.Equal(icicle_runtime.Success, err)
	suite.Equal(test_helpers.MAIN_DEVICE, *activeDevice)
	memory1, err := icicle_runtime.GetAvailableMemory()
	if err == icicle_runtime.ApiNotImplemented {
		suite.T().Skipf("GetAvailableMemory() function is not implemented on %s device", test_helpers.MAIN_DEVICE.GetDeviceType())
	}
	suite.Equal(icicle_runtime.Success, err)
	suite.Greater(memory1.Total, uint(0))
	suite.Greater(memory1.Free, uint(0))
}

type DeviceTestSuite struct {
	suite.Suite
}

func (s *DeviceTestSuite) TestGetDeviceType() {
	s.Run("TestGetDeviceType", test_helpers.TestWrapper(&s.Suite, testGetDeviceType))
	s.Run("TestIsDeviceAvailable", test_helpers.TestWrapper(&s.Suite, testIsDeviceAvailable))
	s.Run("TestSetDefaultDevice", test_helpers.TestWrapper(&s.Suite, testSetDefaultDevice))
	s.Run("TestRegisteredDevices", test_helpers.TestWrapper(&s.Suite, testRegisteredDevices))
	s.Run("TestDeviceProperties", test_helpers.TestWrapper(&s.Suite, testDeviceProperties))
	s.Run("TestActiveDevice", test_helpers.TestWrapper(&s.Suite, testActiveDevice))
}

func TestSuiteDevice(t *testing.T) {
	suite.Run(t, new(DeviceTestSuite))
}
