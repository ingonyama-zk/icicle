package tests

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	icicleGrumpkin "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin/msm"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func testMSM(suite suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6} {
		runtime.SetDevice(&DEVICE)
		size := 1 << power

		scalars := icicleGrumpkin.GenerateScalars(size)
		points := icicleGrumpkin.GenerateAffinePoints(size)

		stream, _ := runtime.CreateStream()
		var p icicleGrumpkin.Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), 1, stream)
		suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")
		cfg.StreamHandle = stream

		e = msm.Msm(scalars, points, &cfg, out)
		suite.Equal(e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[icicleGrumpkin.Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		runtime.SynchronizeStream(stream)
		runtime.DestroyStream(stream)

	}
}

func testMSMBatch(suite suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	for _, power := range []int{5, 6} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleGrumpkin.GenerateScalars(totalSize)
			points := icicleGrumpkin.GenerateAffinePoints(totalSize)

			var p icicleGrumpkin.Projective
			var out core.DeviceSlice
			_, e := out.Malloc(p.Size(), batchSize)
			suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")

			e = msm.Msm(scalars, points, &cfg, out)
			suite.Equal(e, runtime.Success, "Msm failed")
			outHost := make(core.HostSlice[icicleGrumpkin.Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()

		}
	}
}

func testPrecomputePoints(suite suite.Suite) {
	if DEVICE.GetDeviceType() == "CPU" {
		suite.T().Skip("Skipping cpu test")
	}
	cfg := msm.GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor

	for _, power := range []int{7, 8} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)

			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleGrumpkin.GenerateScalars(totalSize)
			points := icicleGrumpkin.GenerateAffinePoints(totalSize)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			suite.Equal(runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			cfg.BatchSize = int32(batchSize)
			cfg.ArePointsSharedInBatch = false
			e = msm.PrecomputeBases(points, &cfg, precomputeOut)
			suite.Equal(runtime.Success, e, "PrecomputeBases failed")

			var p icicleGrumpkin.Projective
			var out core.DeviceSlice
			_, e = out.Malloc(p.Size(), batchSize)
			suite.Equal(runtime.Success, e, "Allocating bytes on device for Projective results failed")

			e = msm.Msm(scalars, precomputeOut, &cfg, out)
			suite.Equal(runtime.Success, e, "Msm failed")
			outHost := make(core.HostSlice[icicleGrumpkin.Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()

		}
	}
}

func testPrecomputePointsSharedBases(suite suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor

	for _, power := range []int{4, 5, 6} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)

			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleGrumpkin.GenerateScalars(totalSize)
			points := icicleGrumpkin.GenerateAffinePoints(size)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			suite.Equal(runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			e = msm.PrecomputeBases(points, &cfg, precomputeOut)
			suite.Equal(runtime.Success, e, "PrecomputeBases failed")

			var p icicleGrumpkin.Projective
			var out core.DeviceSlice
			_, e = out.Malloc(p.Size(), batchSize)
			suite.Equal(runtime.Success, e, "Allocating bytes on device for Projective results failed")

			e = msm.Msm(scalars, precomputeOut, &cfg, out)
			suite.Equal(runtime.Success, e, "Msm failed")
			outHost := make(core.HostSlice[icicleGrumpkin.Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()

		}
	}
}

func testMSMSkewedDistribution(suite suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5} {
		runtime.SetDevice(&DEVICE)

		size := 1 << power

		scalars := icicleGrumpkin.GenerateScalars(size)
		for i := size / 4; i < size; i++ {
			scalars[i].One()
		}
		points := icicleGrumpkin.GenerateAffinePoints(size)
		for i := 0; i < size/4; i++ {
			points[i].Zero()
		}

		var p icicleGrumpkin.Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), 1)
		suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")

		e = msm.Msm(scalars, points, &cfg, out)
		suite.Equal(e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[icicleGrumpkin.Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()

	}
}

func testMSMMultiDevice(suite suite.Suite) {
	numDevices, _ := runtime.GetDeviceCount()
	fmt.Println("There are ", numDevices, " ", DEVICE.GetDeviceType(), " devices available")
	wg := sync.WaitGroup{}

	for i := 0; i < numDevices; i++ {
		currentDevice := runtime.Device{DeviceType: DEVICE.DeviceType, Id: int32(i)}
		wg.Add(1)
		runtime.RunOnDevice(&currentDevice, func(args ...any) {
			defer wg.Done()

			fmt.Println("Running on ", currentDevice.GetDeviceType(), " ", currentDevice.Id, " device")

			cfg := msm.GetDefaultMSMConfig()
			cfg.IsAsync = true
			for _, power := range []int{2, 3, 4, 5, 6} {
				size := 1 << power
				scalars := icicleGrumpkin.GenerateScalars(size)
				points := icicleGrumpkin.GenerateAffinePoints(size)

				stream, _ := runtime.CreateStream()
				var p icicleGrumpkin.Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), 1, stream)
				suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")
				cfg.StreamHandle = stream

				e = msm.Msm(scalars, points, &cfg, out)
				suite.Equal(e, runtime.Success, "Msm failed")
				outHost := make(core.HostSlice[icicleGrumpkin.Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				runtime.SynchronizeStream(stream)
				runtime.DestroyStream(stream)

			}
		})
	}
	wg.Wait()
}

type MSMTestSuite struct {
	suite.Suite
}

func (s *MSMTestSuite) TestMSM() {
	s.Run("TestMSM", testWrapper(s.Suite, testMSM))
	s.Run("TestMSMBatch", testWrapper(s.Suite, testMSMBatch))
	s.Run("TestPrecomputePoints", testWrapper(s.Suite, testPrecomputePoints))
	s.Run("TestPrecomputePointsSharedBases", testWrapper(s.Suite, testPrecomputePointsSharedBases))
	s.Run("TestMSMSkewedDistribution", testWrapper(s.Suite, testMSMSkewedDistribution))
	s.Run("TestMSMMultiDevice", testWrapper(s.Suite, testMSMMultiDevice))
}

func TestSuiteMSM(t *testing.T) {
	suite.Run(t, new(MSMTestSuite))
}
