package tests

import (
	"fmt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
	"os"
	"sync"
	"testing"
)

const (
	largestTestSize = 20
)

var (
	DEVICE   runtime.Device
	exitCode int
)

func initDomain(cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	rouIcicle := babybear.ScalarField{}
	rouIcicle.FromUint32(1461624142)
	e := ntt.InitDomain(rouIcicle, cfg)
	return e
}

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

		// TODO - run tests for each device type without calling `m.Run` multiple times
		// see https://cs.opensource.google/go/go/+/refs/tags/go1.23.1:src/testing/testing.go;l=1936-1940 for more info
		// execute tests
		exitCode |= m.Run()

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
