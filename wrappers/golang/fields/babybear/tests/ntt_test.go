package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testNTTGetDefaultConfig(suite suite.Suite) {
	actual := ntt.GetDefaultNttConfig()
	expected := test_helpers.GenerateLimbOne(int(babybear.SCALAR_LIMBS))
	suite.Equal(expected, actual.CosetGen[:])

	cosetGenField := babybear.ScalarField{}
	cosetGenField.One()
	suite.ElementsMatch(cosetGenField.GetLimbs(), actual.CosetGen)
}

func testNtt(suite suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := babybear.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{4, largestTestSize} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size

			scalarsCopy := core.HostSliceFromElements[babybear.ScalarField](scalars[:testSize])
			cfg.Ordering = v

			// run ntt
			output := make(core.HostSlice[babybear.ScalarField], testSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)

		}
	}
}

func testNttDeviceAsync(suite suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := babybear.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{1, 10, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				runtime.SetDevice(&DEVICE)

				testSize := 1 << size
				scalarsCopy := core.HostSliceFromElements[babybear.ScalarField](scalars[:testSize])

				stream, _ := runtime.CreateStream()

				cfg.Ordering = v
				cfg.IsAsync = true
				cfg.StreamHandle = stream

				var deviceInput core.DeviceSlice
				scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
				var deviceOutput core.DeviceSlice
				deviceOutput.MallocAsync(scalarsCopy.SizeOfElement(), testSize, stream)

				// run ntt
				ntt.Ntt(deviceInput, direction, &cfg, deviceOutput)
				output := make(core.HostSlice[babybear.ScalarField], testSize)
				output.CopyFromDeviceAsync(&deviceOutput, stream)

				runtime.SynchronizeStream(stream)
				runtime.DestroyStream(stream)
			}
		}
	}
}

func testNttBatch(suite suite.Suite) {
	cfg := ntt.GetDefaultNttConfig()
	largestTestSize := 10
	largestBatchSize := 20
	scalars := babybear.GenerateScalars(1 << largestTestSize * largestBatchSize)

	for _, size := range []int{4, largestTestSize} {
		for _, batchSize := range []int{2, 16, largestBatchSize} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size
			totalSize := testSize * batchSize

			scalarsCopy := core.HostSliceFromElements[babybear.ScalarField](scalars[:totalSize])

			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)
			// run ntt
			output := make(core.HostSlice[babybear.ScalarField], totalSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)

		}
	}
}

type NTTTestSuite struct {
	suite.Suite
}

func (s *NTTTestSuite) TestNTT() {
	s.Run("TestNTTGetDefaultConfig", testWrapper(s.Suite, testNTTGetDefaultConfig))
	s.Run("TestNTT", testWrapper(s.Suite, testNtt))
	s.Run("TestNttDeviceAsync", testWrapper(s.Suite, testNttDeviceAsync))
	s.Run("TestNttBatch", testWrapper(s.Suite, testNttBatch))
}

func TestSuiteNTT(t *testing.T) {
	suite.Run(t, new(NTTTestSuite))
}
