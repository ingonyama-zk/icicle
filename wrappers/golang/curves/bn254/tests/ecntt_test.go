//go:build !icicle_exclude_all || ecntt

package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bn254 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	ecntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ecntt"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testECNtt(suite *suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()

	for _, size := range []int{4, 9, 11} {
		for _, dir := range []core.NTTDir{core.KForward, core.KInverse} {
			testSize := 1 << size

			points := bn254.GenerateProjectivePoints(testSize)

			test_helpers.ActivateReferenceDevice()
			output := make(core.HostSlice[bn254.Projective], testSize)
			e := ecntt.ECNtt(points, dir, &cfg, output)
			suite.Equal(runtime.Success, e, "ECNtt failed")

			test_helpers.ActivateMainDevice()
			outputMain := make(core.HostSlice[bn254.Projective], testSize)
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

			outputMainOrig := make(core.HostSlice[bn254.Projective], testSize)
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
	s.Run("TestECNtt", test_helpers.TestWrapper(&s.Suite, testECNtt))
}

func TestSuiteECNtt(t *testing.T) {
	suite.Run(t, new(ECNttTestSuite))
}
