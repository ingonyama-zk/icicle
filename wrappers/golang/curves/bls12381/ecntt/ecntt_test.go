package ecntt

import (
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	icicle_bls12381 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bls12381"

	"github.com/stretchr/testify/assert"
)

const (
	largestTestSize = 17
)

func init() {
	cfg := icicle_bls12381.GetDefaultNttConfig()
	initDomain(largestTestSize, cfg)
}

func initDomain[T any](largestTestSize int, cfg core.NTTConfig[T]) core.IcicleError {
	rouMont, _ := fft.Generator(uint64(1 << largestTestSize))
	rou := rouMont.Bits()
	rouIcicle := icicle_bls12381.ScalarField{}

	rouIcicle.FromLimbs(rou[:])
	e := icicle_bls12381.InitDomain(rouIcicle, cfg.Ctx, false)
	return e
}

func TestECNtt(t *testing.T) {
	cfg := icicle_bls12381.GetDefaultNttConfig()
	points := icicle_bls12381.GenerateProjectivePoints(1 << largestTestSize)

	for _, size := range []int{4, 5, 6, 7, 8} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			testSize := 1 << size

			pointsCopy := core.HostSliceFromElements[icicle_bls12381.Projective](points[:testSize])
			cfg.Ordering = v
			cfg.NttAlgorithm = core.Radix2

			output := make(core.HostSlice[icicle_bls12381.Projective], testSize)
			e := ECNtt(pointsCopy, core.KForward, &cfg, output)
			assert.Equal(t, core.IcicleErrorCode(0), e.IcicleErrorCode, "ECNtt failed")
		}
	}
}
