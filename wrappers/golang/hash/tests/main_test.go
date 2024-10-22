package tests

import (
	"os"
	"sync"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

var (
	devices   []runtime.Device
)

func testWrapper(suite *suite.Suite, fn func(*suite.Suite)) func() {
	return func() {
		wg := sync.WaitGroup{}
		wg.Add(1)
		runtime.RunOnDevice(&devices[0], func(args ...any) {
			defer wg.Done()
			fn(suite)
		})
		wg.Wait()
	}
}

func TestMain(m *testing.M) {
	runtime.LoadBackendFromEnvOrDefault()
	deviceTypes, e := runtime.GetRegisteredDevices()
	if e != runtime.Success {
		panic("Failed to load registered devices")
	}

	for _, deviceType := range deviceTypes {
		device := runtime.CreateDevice(deviceType, 0)
		devices = append([]runtime.Device{device}, devices...)
	}

	os.Exit(m.Run())
}
