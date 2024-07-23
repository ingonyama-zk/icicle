package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	bn254 "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bn254"
	ecntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bn254/ecntt"
	ntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bn254/ntt"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
	"github.com/stretchr/testify/assert"
)

func TestECNtt(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	points := bn254.GenerateProjectivePoints(1 << largestTestSize)

	for _, size := range []int{4, 5, 6, 7, 8} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {

			runtime.SetDevice(&MAIN_DEVICE)

			testSize := 1 << size

			pointsCopy := core.HostSliceFromElements[bn254.Projective](points[:testSize])
			cfg.Ordering = v
			// cfg. = core.Radix2

			output := make(core.HostSlice[bn254.Projective], testSize)
			e := ecntt.ECNtt(pointsCopy, core.KForward, &cfg, output)
			assert.Equal(t, runtime.Success, e, "ECNtt failed")
		}
	}
}
