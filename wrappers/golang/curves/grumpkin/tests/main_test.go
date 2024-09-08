package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

const (
	largestTestSize = 20
)

var DEVICE runtime.Device

func TestMain(m *testing.M) {
	runtime.LoadBackendFromEnvOrDefault()
	devices, e := runtime.GetRegisteredDevices()
	if e != runtime.Success {
		panic("Failed to load registered devices")
	}
	for _, deviceType := range devices {
		DEVICE = runtime.CreateDevice(deviceType, 0)
		runtime.SetDevice(&DEVICE)

		// execute tests
		m.Run()

	}
}
