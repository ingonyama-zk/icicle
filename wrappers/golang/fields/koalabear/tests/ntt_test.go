//go:build !icicle_exclude_all || ntt

package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	koalabear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testNTTGetDefaultConfig(suite *suite.Suite) {
	actual := ntt.GetDefaultNttConfig()
	expected := test_helpers.GenerateLimbOne(int(koalabear.SCALAR_LIMBS))
	suite.Equal(expected, actual.CosetGen[:])

	cosetGenField := koalabear.ScalarField{}
	cosetGenField.One()
	suite.ElementsMatch(cosetGenField.GetLimbs(), actual.CosetGen)
}

func testNtt(suite *suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := koalabear.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{4, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				testSize := 1 << size

				scalarsCopy := core.HostSliceFromElements[koalabear.ScalarField](scalars[:testSize])
				cfg.Ordering = v

				// run ntt
				test_helpers.ActivateReferenceDevice()
				output := make(core.HostSlice[koalabear.ScalarField], testSize)
				ntt.Ntt(scalarsCopy, direction, &cfg, output)

				test_helpers.ActivateMainDevice()
				outputMain := make(core.HostSlice[koalabear.ScalarField], testSize)
				ntt.Ntt(scalarsCopy, direction, &cfg, outputMain)

				suite.Equal(output, outputMain, "NTT Failed")
			}
		}
	}
}

func testNttDeviceAsync(suite *suite.Suite) {
	scalars := koalabear.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{1, 10, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				testSize := 1 << size
				scalarsCopy := core.HostSliceFromElements[koalabear.ScalarField](scalars[:testSize])

				// Ref device
				test_helpers.ActivateReferenceDevice()
				cfg := ntt.GetDefaultNttConfig()
				cfg.Ordering = v
				cfg.IsAsync = true
				stream, _ := runtime.CreateStream()
				cfg.StreamHandle = stream

				var deviceInput core.DeviceSlice
				scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
				var deviceOutput core.DeviceSlice
				deviceOutput.MallocAsync(scalarsCopy.SizeOfElement(), testSize, stream)

				// run ntt
				ntt.Ntt(deviceInput, direction, &cfg, deviceOutput)
				output := make(core.HostSlice[koalabear.ScalarField], testSize)
				output.CopyFromDeviceAsync(&deviceOutput, stream)

				runtime.SynchronizeStream(stream)
				runtime.DestroyStream(stream)

				// Main device
				test_helpers.ActivateMainDevice()
				cfgMain := ntt.GetDefaultNttConfig()
				cfgMain.Ordering = v
				cfgMain.IsAsync = true
				streamMain, _ := runtime.CreateStream()
				cfgMain.StreamHandle = streamMain

				var deviceInputMain core.DeviceSlice
				scalarsCopy.CopyToDeviceAsync(&deviceInputMain, streamMain, true)
				var deviceOutputMain core.DeviceSlice
				deviceOutputMain.MallocAsync(scalarsCopy.SizeOfElement(), testSize, streamMain)

				// run ntt
				ntt.Ntt(deviceInputMain, direction, &cfgMain, deviceOutputMain)
				outputMain := make(core.HostSlice[koalabear.ScalarField], testSize)
				outputMain.CopyFromDeviceAsync(&deviceOutputMain, streamMain)

				runtime.SynchronizeStream(streamMain)
				runtime.DestroyStream(streamMain)

				suite.Equal(output, outputMain, "NTT DeviceSlice async failed")
			}
		}
	}
}

func testNttBatch(suite *suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()
	largestTestSize := 10
	largestBatchSize := 20
	scalars := koalabear.GenerateScalars(1 << largestTestSize * largestBatchSize)

	for _, size := range []int{4, largestTestSize} {
		for _, batchSize := range []int{2, 16, largestBatchSize} {
			testSize := 1 << size
			totalSize := testSize * batchSize

			scalarsCopy := core.HostSliceFromElements[koalabear.ScalarField](scalars[:totalSize])

			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)

			// Ref device
			test_helpers.ActivateReferenceDevice()
			output := make(core.HostSlice[koalabear.ScalarField], totalSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)

			// Main device
			test_helpers.ActivateMainDevice()
			outputMain := make(core.HostSlice[koalabear.ScalarField], totalSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, outputMain)

			suite.Equal(output, outputMain, "Ntt Batch failed")
		}
	}
}

func testGetRootOfUnity(suite *suite.Suite) {
	n := uint64(1 << largestTestSize)
	rou := ntt.GetRootOfUnity(n)
	suite.NotNil(rou, "GetRootOfUnity returned nil")
}

type NTTTestSuite struct {
	suite.Suite
}

func (s *NTTTestSuite) TestNTT() {
	s.Run("TestNTTGetDefaultConfig", test_helpers.TestWrapper(&s.Suite, testNTTGetDefaultConfig))
	s.Run("TestNTT", test_helpers.TestWrapper(&s.Suite, testNtt))
	s.Run("TestNttDeviceAsync", test_helpers.TestWrapper(&s.Suite, testNttDeviceAsync))
	s.Run("TestNttBatch", test_helpers.TestWrapper(&s.Suite, testNttBatch))
	s.Run("TestGetRootOfUnity", test_helpers.TestWrapper(&s.Suite, testGetRootOfUnity))
}

func TestSuiteNTT(t *testing.T) {
	suite.Run(t, new(NTTTestSuite))
}
