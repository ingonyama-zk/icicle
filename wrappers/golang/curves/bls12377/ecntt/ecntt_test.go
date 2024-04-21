package ecntt

import (
	"os"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bls12-377/fr/fft"
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bls12377"
	ntt "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bls12377/ntt"
	"github.com/stretchr/testify/assert"
)

const (
	largestTestSize = 17
)

func TestECNtt(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	points := bls12_377.GenerateProjectivePoints(1 << largestTestSize)

	for _, size := range []int{4, 5, 6, 7, 8} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			testSize := 1 << size

			pointsCopy := core.HostSliceFromElements[bls12_377.Projective](points[:testSize])
			cfg.Ordering = v
			cfg.NttAlgorithm = core.Radix2

			output := make(core.HostSlice[bls12_377.Projective], testSize)
			e := ECNtt(pointsCopy, core.KForward, &cfg, output)
			assert.Equal(t, core.IcicleErrorCode(0), e.IcicleErrorCode, "ECNtt failed")
		}
	}
}

func TestMain(m *testing.M) {
	// setup domain
	cfg := ntt.GetDefaultNttConfig()
	rouMont, _ := fft.Generator(uint64(1 << largestTestSize))
	rou := rouMont.Bits()
	rouIcicle := bls12_377.ScalarField{}

	rouIcicle.FromLimbs(core.ConvertUint64ArrToUint32Arr(rou[:]))
	e := ntt.InitDomain(rouIcicle, cfg.Ctx, false)
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
