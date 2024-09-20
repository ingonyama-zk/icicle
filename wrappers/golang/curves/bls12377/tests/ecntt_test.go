package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"
	ecntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/ecntt"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"
	"github.com/stretchr/testify/suite"
)

func testECNtt(suite suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()
	ext := config_extension.Create()
	ext.SetInt(core.CUDA_NTT_ALGORITHM, int(core.Radix2))
	cfg.Ext = ext.AsUnsafePointer()

	points := bls12_377.GenerateProjectivePoints(1 << largestTestSize)

	for _, size := range []int{4, 5, 6, 7, 8} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size

			pointsCopy := core.HostSliceFromElements[bls12_377.Projective](points[:testSize])
			cfg.Ordering = v

			output := make(core.HostSlice[bls12_377.Projective], testSize)
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
