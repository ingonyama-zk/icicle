package ntt

import (
	"os"
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	babybear "github.com/ingonyama-zk/icicle/wrappers/golang/fields/babybear"
	babybear_extension "github.com/ingonyama-zk/icicle/wrappers/golang/fields/babybear/extension"
	ntt "github.com/ingonyama-zk/icicle/wrappers/golang/fields/babybear/ntt"
)

const (
	largestTestSize = 17
)

func initDomain[T any](largestTestSize int, cfg core.NTTConfig[T]) core.IcicleError {
	rouIcicle := babybear.ScalarField{}
	rouIcicle.FromUint32(1461624142)
	e := ntt.InitDomain(rouIcicle, cfg.Ctx, false)
	return e
}

func TestNtt(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := babybear_extension.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{4, largestTestSize} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			testSize := 1 << size

			scalarsCopy := core.HostSliceFromElements[babybear_extension.ExtensionField](scalars[:testSize])
			cfg.Ordering = v

			// run ntt
			output := make(core.HostSlice[babybear_extension.ExtensionField], testSize)
			Ntt(scalarsCopy, core.KForward, &cfg, output)
		}
	}
}

func TestNttDeviceAsync(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := babybear_extension.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{1, 10, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				testSize := 1 << size
				scalarsCopy := core.HostSliceFromElements[babybear_extension.ExtensionField](scalars[:testSize])

				stream, _ := cr.CreateStream()

				cfg.Ordering = v
				cfg.IsAsync = true
				cfg.Ctx.Stream = &stream

				var deviceInput core.DeviceSlice
				scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
				var deviceOutput core.DeviceSlice
				deviceOutput.MallocAsync(testSize*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

				// run ntt
				Ntt(deviceInput, direction, &cfg, deviceOutput)
				output := make(core.HostSlice[babybear_extension.ExtensionField], testSize)
				output.CopyFromDeviceAsync(&deviceOutput, stream)

				cr.SynchronizeStream(&stream)
			}
		}
	}
}

func TestNttBatch(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	largestBatchSize := 100
	scalars := babybear_extension.GenerateScalars(1 << largestTestSize * largestBatchSize)

	for _, size := range []int{4, largestTestSize} {
		for _, batchSize := range []int{1, 16, largestBatchSize} {
			testSize := 1 << size
			totalSize := testSize * batchSize

			scalarsCopy := core.HostSliceFromElements[babybear_extension.ExtensionField](scalars[:totalSize])

			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)
			// run ntt
			output := make(core.HostSlice[babybear_extension.ExtensionField], totalSize)
			Ntt(scalarsCopy, core.KForward, &cfg, output)
		}
	}
}

func TestMain(m *testing.M) {
	// setup domain
	cfg := ntt.GetDefaultNttConfig()
	e := initDomain(largestTestSize, cfg)
	if e.IcicleErrorCode != core.IcicleErrorCode(0) {
		panic("initDomain failed")
	}

	// execute tests
	os.Exit(m.Run())

	// release domain
	e = ntt.ReleaseDomain(cfg.Ctx)
	if e.IcicleErrorCode != core.IcicleErrorCode(0) {
		panic("ReleaseDomain failed")
	}
}
