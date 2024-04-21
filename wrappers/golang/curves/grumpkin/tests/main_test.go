package tests

import (
	"os"
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	grumpkin "github.com/ingonyama-zk/icicle/wrappers/golang/curves/grumpkin"
	ntt "github.com/ingonyama-zk/icicle/wrappers/golang/curves/grumpkin/ntt"
	poly "github.com/ingonyama-zk/icicle/wrappers/golang/curves/grumpkin/polynomial"
)

const (
	largestTestSize = 20
)

func initDomain[T any](largestTestSize int, cfg core.NTTConfig[T]) core.IcicleError {
	rouIcicle := grumpkin.ScalarField{}
	rouIcicle.FromUint32(0)
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
