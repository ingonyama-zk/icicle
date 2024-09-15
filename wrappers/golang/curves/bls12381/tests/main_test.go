package tests

import (
	"fmt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_381 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
	"sync"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
)

const (
	largestTestSize = 20
)

var DEVICE runtime.Device

func initDomain(largestTestSize int, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	rouMont, _ := fft.Generator(uint64(1 << largestTestSize))
	rou := rouMont.Bits()
	rouIcicle := bls12_381.ScalarField{}
	limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

	rouIcicle.FromLimbs(limbs)
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
		e = initDomain(largestTestSize, cfg)
		if e != runtime.Success {
			if e != runtime.ApiNotImplemented {
				fmt.Println("initDomain is not implemented for ", deviceType, " device type")
			} else {
				panic("initDomain failed")
			}
		}

		// execute tests
		m.Run()

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
}
