package tests

import (
	"fmt"
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	bn254 "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bn254"
	ntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bn254/ntt"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

const (
	largestTestSize = 20
)

var DEVICE runtime.Device

func initDomain(largestTestSize int, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	rouMont, _ := fft.Generator(uint64(1 << largestTestSize))
	rou := rouMont.Bits()
	rouIcicle := bn254.ScalarField{}
	limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

	rouIcicle.FromLimbs(limbs)
	e := ntt.InitDomain(rouIcicle, cfg)
	return e
}

func TestMain(m *testing.M) {
	runtime.LoadBackendFromEnv()
	devices, e := runtime.GetRegisteredDevices()
	if e != runtime.Success {
		panic("Failed to load registered devices")
	}
	for _, deviceType := range devices {
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
