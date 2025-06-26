//go:build linux

package tests

import (
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
