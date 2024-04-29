package tests

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	icicleGrumpkin "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/grumpkin"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/grumpkin/msm"
)

func TestMSM(t *testing.T) {
	cfg := msm.GetDefaultMSMConfigForDevice(0)
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := icicleGrumpkin.GenerateScalars(size)
		points := icicleGrumpkin.GenerateAffinePoints(size)

		cr.SetDevice(cfg.Ctx.GetDeviceId())
		stream, _ := cr.CreateStream()
		var p icicleGrumpkin.Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), p.Size(), stream)
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
		cfg.Ctx.Stream = &stream

		e = msm.Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[icicleGrumpkin.Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		cr.SynchronizeStream(&stream)

	}
}

func TestMSMBatch(t *testing.T) {
	cfg := msm.GetDefaultMSMConfigForDevice(0)
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleGrumpkin.GenerateScalars(totalSize)
			points := icicleGrumpkin.GenerateAffinePoints(totalSize)

			var p icicleGrumpkin.Projective
			var out core.DeviceSlice
			_, e := out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			cr.SetDevice(cfg.Ctx.GetDeviceId())
			e = msm.Msm(scalars, points, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[icicleGrumpkin.Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()

		}
	}
}

func TestPrecomputeBase(t *testing.T) {
	cfg := msm.GetDefaultMSMConfigForDevice(0)
	const precomputeFactor = 8
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleGrumpkin.GenerateScalars(totalSize)
			points := icicleGrumpkin.GenerateAffinePoints(totalSize)

			cr.SetDevice(cfg.Ctx.GetDeviceId())
			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for PrecomputeBases results failed")

			e = msm.PrecomputeBases(points, precomputeFactor, 0, &cfg.Ctx, precomputeOut)
			assert.Equal(t, e, cr.CudaSuccess, "PrecomputeBases failed")

			var p icicleGrumpkin.Projective
			var out core.DeviceSlice
			_, e = out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			cfg.PrecomputeFactor = precomputeFactor

			e = msm.Msm(scalars, precomputeOut, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[icicleGrumpkin.Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()

		}
	}
}

func TestMSMSkewedDistribution(t *testing.T) {
	cfg := msm.GetDefaultMSMConfigForDevice(0)
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := icicleGrumpkin.GenerateScalars(size)
		for i := size / 4; i < size; i++ {
			scalars[i].One()
		}
		points := icicleGrumpkin.GenerateAffinePoints(size)
		for i := 0; i < size/4; i++ {
			points[i].Zero()
		}

		cr.SetDevice(cfg.Ctx.GetDeviceId())
		var p icicleGrumpkin.Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

		e = msm.Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[icicleGrumpkin.Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()

	}
}

func TestMSMMultiDevice(t *testing.T) {
	numDevices, _ := cr.GetDeviceCount()
	fmt.Println("There are ", numDevices, " devices available")
	orig_device, _ := cr.GetDevice()
	wg := sync.WaitGroup{}

	for i := 0; i < numDevices; i++ {
		wg.Add(1)
		cr.RunOnDevice(i, func(args ...any) {
			defer wg.Done()
			cfg := msm.GetDefaultMSMConfig()
			cfg.IsAsync = true
			for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
				size := 1 << power
				scalars := icicleGrumpkin.GenerateScalars(size)
				points := icicleGrumpkin.GenerateAffinePoints(size)

				stream, _ := cr.CreateStream()
				var p icicleGrumpkin.Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), p.Size(), stream)
				assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
				cfg.Ctx.Stream = &stream

				e = msm.Msm(scalars, points, &cfg, out)
				assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
				outHost := make(core.HostSlice[icicleGrumpkin.Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				cr.SynchronizeStream(&stream)

			}
		})
	}
	wg.Wait()
	cr.SetDevice(orig_device)
}
