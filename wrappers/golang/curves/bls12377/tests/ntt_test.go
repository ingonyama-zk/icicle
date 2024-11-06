package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testNTTGetDefaultConfig(suite *suite.Suite) {
	actual := ntt.GetDefaultNttConfig()
	expected := test_helpers.GenerateLimbOne(int(bls12_377.SCALAR_LIMBS))
	suite.Equal(expected, actual.CosetGen[:])

	cosetGenField := bls12_377.ScalarField{}
	cosetGenField.One()
	suite.ElementsMatch(cosetGenField.GetLimbs(), actual.CosetGen)
}

func testNtt(suite *suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := bls12_377.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{4, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				testSize := 1 << size

				scalarsCopy := core.HostSliceFromElements[bls12_377.ScalarField](scalars[:testSize])
				cfg.Ordering = v

				// run ntt
				test_helpers.ActivateReferenceDevice()
				output := make(core.HostSlice[bls12_377.ScalarField], testSize)
				ntt.Ntt(scalarsCopy, direction, &cfg, output)

				test_helpers.ActivateMainDevice()
				outputMain := make(core.HostSlice[bls12_377.ScalarField], testSize)
				ntt.Ntt(scalarsCopy, direction, &cfg, outputMain)

				suite.Equal(output, outputMain, "NTT Failed")
			}
		}
	}
}

func testNttDeviceAsync(suite *suite.Suite) {
	scalars := bls12_377.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{1, 10, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				testSize := 1 << size
				scalarsCopy := core.HostSliceFromElements[bls12_377.ScalarField](scalars[:testSize])

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
				output := make(core.HostSlice[bls12_377.ScalarField], testSize)
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
				outputMain := make(core.HostSlice[bls12_377.ScalarField], testSize)
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
	scalars := bls12_377.GenerateScalars(1 << largestTestSize * largestBatchSize)

	for _, size := range []int{4, largestTestSize} {
		for _, batchSize := range []int{2, 16, largestBatchSize} {
			testSize := 1 << size
			totalSize := testSize * batchSize

			scalarsCopy := core.HostSliceFromElements[bls12_377.ScalarField](scalars[:totalSize])

			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)

			// Ref device
			test_helpers.ActivateReferenceDevice()
			output := make(core.HostSlice[bls12_377.ScalarField], totalSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)

			// Main device
			test_helpers.ActivateMainDevice()
			outputMain := make(core.HostSlice[bls12_377.ScalarField], totalSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, outputMain)

			suite.Equal(output, outputMain, "Ntt Batch failed")
		}
	}
}

type NTTTestSuite struct {
	suite.Suite
}

func (s *NTTTestSuite) TestNTT() {
	s.Run("TestNTTGetDefaultConfig", testWrapper(&s.Suite, testNTTGetDefaultConfig))
	s.Run("TestNTT", testWrapper(&s.Suite, testNtt))
	s.Run("TestNttDeviceAsync", testWrapper(&s.Suite, testNttDeviceAsync))
	s.Run("TestNttBatch", testWrapper(&s.Suite, testNttBatch))
}

func TestSuiteNTT(t *testing.T) {
	suite.Run(t, new(NTTTestSuite))
}
