package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear_extension "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/extension"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func TestNttNoDomain(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := babybear_extension.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{4, largestTestSize} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size

			scalarsCopy := core.HostSliceFromElements[babybear_extension.ExtensionField](scalars[:testSize])
			cfg.Ordering = v

			// run ntt
			output := make(core.HostSlice[babybear_extension.ExtensionField], testSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)
		}
	}
}

func TestNttDeviceAsyncNoDomain(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := babybear_extension.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{1, 10, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				runtime.SetDevice(&DEVICE)

				testSize := 1 << size
				scalarsCopy := core.HostSliceFromElements[babybear_extension.ExtensionField](scalars[:testSize])

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
				output := make(core.HostSlice[babybear_extension.ExtensionField], testSize)
				output.CopyFromDeviceAsync(&deviceOutput, stream)

				runtime.SynchronizeStream(stream)
			}
		}
	}
}

func TestNttBatchNoDomain(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	largestTestSize := 12
	largestBatchSize := 100
	scalars := babybear_extension.GenerateScalars(1 << largestTestSize * largestBatchSize)

	for _, size := range []int{4, largestTestSize} {
		for _, batchSize := range []int{2, 16, largestBatchSize} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size
			totalSize := testSize * batchSize

			scalarsCopy := core.HostSliceFromElements[babybear_extension.ExtensionField](scalars[:totalSize])

			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)
			// run ntt
			output := make(core.HostSlice[babybear_extension.ExtensionField], totalSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)
		}
	}
}
