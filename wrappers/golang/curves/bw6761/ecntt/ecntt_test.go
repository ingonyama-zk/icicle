package ecntt

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	. "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bw6761"

	"github.com/stretchr/testify/assert"
)

const (
	largestTestSize = 17
)

func TestECNtt(t *testing.T) {
	cfg := GetDefaultNttConfig()
	points := GenerateProjectivePoints(1 << largestTestSize)

	for _, size := range []int{4, 5, 6, 7, 8} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			testSize := 1 << size

			pointsCopy := core.HostSliceFromElements[Projective](points[:testSize])
			cfg.Ordering = v
			cfg.NttAlgorithm = core.Radix2

			output := make(core.HostSlice[Projective], testSize)
			e := ECNtt(pointsCopy, core.KForward, &cfg, output)
			assert.Equal(t, core.IcicleErrorCode(0), e.IcicleErrorCode, "ECNtt failed")
		}
	}
}
