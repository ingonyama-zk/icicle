package tests

import (
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

const (
	largestTestSize = 20
)

func initDomain(cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	rouIcicle := babybear.ScalarField{}
	rouIcicle.FromUint32(1461624142)
	e := ntt.InitDomain(rouIcicle, cfg)
	return e
}

func testWrapper(suite *suite.Suite, fn func(*suite.Suite)) func() {
	return func() {
		wg := sync.WaitGroup{}
		wg.Add(1)
		runtime.RunOnDevice(&test_helpers.REFERENCE_DEVICE, func(args ...any) {
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
		device := runtime.CreateDevice(deviceType, 0)
		runtime.SetDevice(&device)

		// setup domain
		cfg := core.GetDefaultNTTInitDomainConfig()
		e = initDomain(cfg)
		if e != runtime.Success {
			if e != runtime.ApiNotImplemented {
				fmt.Println("initDomain is not implemented for ", deviceType, " device type")
			} else {
				panic("initDomain failed")
			}
		}
	}

	exitCode := m.Run()

	// release domain
	for _, deviceType := range devices {
		device := runtime.CreateDevice(deviceType, 0)
		runtime.SetDevice(&device)
		// release domain
		e = ntt.ReleaseDomain()
		if e != runtime.Success {
			if e != runtime.ApiNotImplemented {
				fmt.Println("ReleaseDomain is not implemented for ", deviceType, " device type")
			} else {
				panic("ReleaseDomain failed")
			}
		}
	}

	os.Exit(exitCode)
}
