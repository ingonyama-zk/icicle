package tests

import (
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	icicle_runtime "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/stretchr/testify/suite"
)

func testIsDeviceAvailable(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()
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
		suite.T().Skip("Failed to get number of devices:", wcErr)
	}

	suite.Equal(icicle_runtime.Success, err)
	suite.Equal(expectedNumDevices, res)

	suite.Equal(icicle_runtime.Success, err)
	suite.True(icicle_runtime.IsDeviceAvailable(&test_helpers.MAIN_DEVICE))
	suite.True(icicle_runtime.IsDeviceAvailable(&test_helpers.REFERENCE_DEVICE))
	devInvalid := icicle_runtime.CreateDevice("invalid", 0)
	suite.False(icicle_runtime.IsDeviceAvailable(&devInvalid))
}

func testSetDefaultDevice(suite *suite.Suite) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	defaultDevice := test_helpers.MAIN_DEVICE
	icicle_runtime.SetDefaultDevice(&defaultDevice)

	outerThreadID := syscall.Gettid()
	done := make(chan struct{}, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		// Ensure we are operating on an OS thread other than the original one
		tidInner := syscall.Gettid()
		for tidInner == outerThreadID {
			fmt.Println("Locked thread is the same as original, getting new locked thread")
			runtime.UnlockOSThread()
			runtime.LockOSThread()
			tidInner = syscall.Gettid()
		}

		activeDevice, err := icicle_runtime.GetActiveDevice()
		suite.Equal(icicle_runtime.Success, err)
		suite.Equal(defaultDevice, *activeDevice)

		close(done)
	}()

	<-done

	icicle_runtime.SetDefaultDevice(&test_helpers.REFERENCE_DEVICE)
}
