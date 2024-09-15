package tests

import (
	"fmt"
	"github.com/stretchr/testify/suite"
	"sync"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

const (
	largestTestSize = 20
)

var DEVICE runtime.Device

func testWrapper(suite suite.Suite, fn func(suite.Suite)) func() {
	return func() {
		wg := sync.WaitGroup{}
		wg.Add(1)
		runtime.RunOnDevice(&DEVICE, func(args ...any) {
			defer wg.Done()
			fn(suite)
		})
		wg.Wait()
	}
}

func TestMain(m *testing.M) {
	runtime.LoadBackendFromEnvOrDefault()
	devices, e := runtime.GetRegisteredDevices()
	if e != runtime.Success {
		panic("Failed to load registered devices")
	}
	for _, deviceType := range devices {
		fmt.Println("Running tests for device type:", deviceType)
		DEVICE = runtime.CreateDevice(deviceType, 0)
		runtime.SetDevice(&DEVICE)

		// execute tests
		m.Run()

	}
}
