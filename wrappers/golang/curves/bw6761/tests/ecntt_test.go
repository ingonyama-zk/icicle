package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bw6_761 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761"
	ecntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761/ecntt"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testECNtt(suite *suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()

	for _, size := range []int{4, 9, 11} {
		for _, dir := range []core.NTTDir{core.KForward, core.KInverse} {
			testSize := 1 << size

			points := bw6_761.GenerateProjectivePoints(testSize)

			test_helpers.ActivateReferenceDevice()
			output := make(core.HostSlice[bw6_761.Projective], testSize)
			e := ecntt.ECNtt(points, dir, &cfg, output)
			suite.Equal(runtime.Success, e, "ECNtt failed")

			test_helpers.ActivateMainDevice()
			outputMain := make(core.HostSlice[bw6_761.Projective], testSize)
			eMain := ecntt.ECNtt(points, dir, &cfg, outputMain)
			suite.Equal(runtime.Success, eMain, "ECNtt failed")

			for i := 0; i < testSize; i++ {
				suite.True(output[i].ProjectiveEq(&outputMain[i]), "Failed at index: ", i)
			}

			if dir == core.KForward {
				dir = core.KInverse
			} else {
				dir = core.KForward
			}

			outputMainOrig := make(core.HostSlice[bw6_761.Projective], testSize)
			eMain = ecntt.ECNtt(outputMain, dir, &cfg, outputMainOrig)
			suite.Equal(runtime.Success, eMain, "ECNtt failed")

			for i := 0; i < testSize; i++ {
				suite.True(outputMainOrig[i].ProjectiveEq(&points[i]), "Failed at index: ", i)
			}
		}
	}
}

type ECNttTestSuite struct {
	suite.Suite
}

func (s *ECNttTestSuite) TestECNtt() {
	s.Run("TestECNtt", testWrapper(&s.Suite, testECNtt))
}

func TestSuiteECNtt(t *testing.T) {
	suite.Run(t, new(ECNttTestSuite))
}
