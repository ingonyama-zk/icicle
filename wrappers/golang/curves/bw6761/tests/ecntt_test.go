package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bw6_761 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761"
	ecntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761/ecntt"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"
	"github.com/stretchr/testify/suite"
)

func testECNtt(suite suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()
	ext := config_extension.Create()
	ext.SetInt(core.CUDA_NTT_ALGORITHM, int(core.Radix2))
	cfg.Ext = ext.AsUnsafePointer()

	points := bw6_761.GenerateProjectivePoints(1 << largestTestSize)

	for _, size := range []int{4, 5, 6, 7, 8} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size

			pointsCopy := core.HostSliceFromElements[bw6_761.Projective](points[:testSize])
			cfg.Ordering = v

			output := make(core.HostSlice[bw6_761.Projective], testSize)
			e := ecntt.ECNtt(pointsCopy, core.KForward, &cfg, output)
			suite.Equal(runtime.Success, e, "ECNtt failed")
		}
	}
}

type ECNttTestSuite struct {
	suite.Suite
}

func (s *ECNttTestSuite) TestECNtt() {
	s.Run("TestECNtt", testWrapper(s.Suite, testECNtt))
}

func TestSuiteECNtt(t *testing.T) {
	suite.Run(t, new(ECNttTestSuite))
}
