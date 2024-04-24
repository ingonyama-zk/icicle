package tests

import (
	"os"
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	bw6_761 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bw6761"
	ntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bw6761/ntt"
	poly "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bw6761/polynomial"

	"github.com/consensys/gnark-crypto/ecc/bw6-761/fr/fft"
)

const (
	largestTestSize = 20
)

func initDomain[T any](largestTestSize int, cfg core.NTTConfig[T]) core.IcicleError {
	rouMont, _ := fft.Generator(uint64(1 << largestTestSize))
	rou := rouMont.Bits()
	rouIcicle := bw6_761.ScalarField{}
	limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

	rouIcicle.FromLimbs(limbs)
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
