//go:build ntt

package tests

import (
	"fmt"
	"os"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_381 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

const (
	largestTestSize = 20
)

func initDomain(largestTestSize int, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	rouMont, _ := fft.Generator(uint64(1 << largestTestSize))
	rou := rouMont.Bits()
	rouIcicle := bls12_381.ScalarField{}
	limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

	rouIcicle.FromLimbs(limbs)
	e := ntt.InitDomain(rouIcicle, cfg)
	return e
}

func TestMain(m *testing.M) {
	test_helpers.LOAD_AND_INIT_MAIN_DEVICE()
	devices, e := runtime.GetRegisteredDevices()
	if e != runtime.Success {
		panic("Failed to load registered devices")
	}
	for _, deviceType := range devices {
		device := runtime.CreateDevice(deviceType, 0)
		runtime.SetDevice(&device)

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
