package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
)

func TestNTTGetDefaultConfig(t *testing.T) {
	actual := ntt.GetDefaultNttConfig()
	expected := test_helpers.GenerateLimbOne(int(babybear.SCALAR_LIMBS))
	assert.Equal(t, expected, actual.CosetGen[:])

	cosetGenField := babybear.ScalarField{}
	cosetGenField.One()
	assert.ElementsMatch(t, cosetGenField.GetLimbs(), actual.CosetGen)
}

func TestInitDomain(t *testing.T) {
	t.Skip("Skipped because each test requires the domain to be initialized before running. We ensure this using the TestMain() function")
	cfg := core.GetDefaultNTTInitDomainConfig()
	assert.NotPanics(t, func() { initDomain(largestTestSize, cfg) })
}

func TestNtt(t *testing.T) {
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

func TestNttDeviceAsync(t *testing.T) {
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
			}
		}
	}
}

func TestNttBatch(t *testing.T) {
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

func TestReleaseDomain(t *testing.T) {
	t.Skip("Skipped because each test requires the domain to be initialized before running. We ensure this using the TestMain() function")
	e := ntt.ReleaseDomain()
	assert.Equal(t, runtime.Success, e, "ReleasDomain failed")
}
