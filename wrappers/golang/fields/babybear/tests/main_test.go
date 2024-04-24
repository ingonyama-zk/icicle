package tests

import (
	"os"
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	babybear "github.com/ingonyama-zk/icicle/v2/wrappers/golang/fields/babybear"
	ntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang/fields/babybear/ntt"
	poly "github.com/ingonyama-zk/icicle/v2/wrappers/golang/fields/babybear/polynomial"
)

const (
	largestTestSize = 20
)

func initDomain[T any](largestTestSize int, cfg core.NTTConfig[T]) core.IcicleError {
	rouIcicle := babybear.ScalarField{}
	rouIcicle.FromUint32(1461624142)
	e := ntt.InitDomain(rouIcicle, cfg.Ctx, false)
	return e
}

func TestMain(m *testing.M) {
	poly.InitPolyBackend()

	// setup domain
	cfg := ntt.GetDefaultNttConfig()
	e := initDomain(largestTestSize, cfg)
	if e.IcicleErrorCode != core.IcicleErrorCode(0) {
		panic("initDomain failed")
	}

	// execute tests
	os.Exit(m.Run())

	// release domain
	e = ntt.ReleaseDomain(cfg.Ctx)
	if e.IcicleErrorCode != core.IcicleErrorCode(0) {
		panic("ReleaseDomain failed")
	}
}
