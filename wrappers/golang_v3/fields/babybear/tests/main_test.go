package tests

import (
	"os"
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	babybear "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/fields/babybear"
	ntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/fields/babybear/ntt"

	// poly "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/fields/babybear/polynomial"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
)

const (
	largestTestSize = 20
)

func initDomain(largestTestSize int, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	rouIcicle := babybear.ScalarField{}
	rouIcicle.FromUint32(1461624142)
	e := ntt.InitDomain(rouIcicle, cfg)
	return e
}

func TestMain(m *testing.M) {
	runtime.LoadBackendFromEnv()
	device := runtime.CreateDevice("CUDA", 0)
	runtime.SetDevice(&device)

	// setup domain
	cfg := core.GetDefaultNTTInitDomainConfig()
	e := initDomain(largestTestSize, cfg)
	if e != runtime.Success {
		panic("initDomain failed")
	}

	// execute tests
	os.Exit(m.Run())

	// release domain
	e = ntt.ReleaseDomain()
	if e != runtime.Success {
		panic("ReleaseDomain failed")
	}
}
