package tests

import (
	"sync"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	icicleBls12_381 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/msm"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testMSM(suite *suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6} {
		size := 1 << power

		scalars := icicleBls12_381.GenerateScalars(size)
		points := icicleBls12_381.GenerateAffinePoints(size)

		// CPU run
		test_helpers.ActivateReferenceDevice()
		stream, _ := runtime.CreateStream()
		var p icicleBls12_381.Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), 1, stream)
		suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")
		cfg.StreamHandle = stream

		e = msm.Msm(scalars, points, &cfg, out)
		suite.Equal(e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[icicleBls12_381.Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		runtime.SynchronizeStream(stream)
		runtime.DestroyStream(stream)

		// Cuda run
		test_helpers.ActivateMainDevice()
		streamMain, _ := runtime.CreateStream()
		var pMain icicleBls12_381.Projective
		var outMain core.DeviceSlice
		_, eMain := outMain.MallocAsync(pMain.Size(), 1, streamMain)
		suite.Equal(eMain, runtime.Success, "Allocating bytes on device for Projective results failed")
		cfg.StreamHandle = stream

		e = msm.Msm(scalars, points, &cfg, outMain)
		suite.Equal(e, runtime.Success, "Msm failed")
		outHostMain := make(core.HostSlice[icicleBls12_381.Projective], 1)
		outHostMain.CopyFromDeviceAsync(&outMain, streamMain)
		outMain.FreeAsync(streamMain)

		runtime.SynchronizeStream(streamMain)
		runtime.DestroyStream(streamMain)

		suite.Equal(out, outMain)
	}
}

func testMSMBatch(suite *suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	for _, power := range []int{5, 6} {
		for _, batchSize := range []int{1, 3, 5} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_381.GenerateScalars(totalSize)
			points := icicleBls12_381.GenerateAffinePoints(totalSize)

			test_helpers.ActivateReferenceDevice()
			var p icicleBls12_381.Projective
			var out core.DeviceSlice
			_, e := out.Malloc(p.Size(), batchSize)
			suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")

			e = msm.Msm(scalars, points, &cfg, out)
			suite.Equal(e, runtime.Success, "Msm failed")
			outHost := make(core.HostSlice[icicleBls12_381.Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()

			test_helpers.ActivateMainDevice()
			var pMain icicleBls12_381.Projective
			var outMain core.DeviceSlice
			_, eMain := outMain.Malloc(pMain.Size(), batchSize)
			suite.Equal(eMain, runtime.Success, "Allocating bytes on device for Projective results failed")

			eMain = msm.Msm(scalars, points, &cfg, outMain)
			suite.Equal(eMain, runtime.Success, "Msm failed")
			outHostMain := make(core.HostSlice[icicleBls12_381.Projective], batchSize)
			outHostMain.CopyFromDevice(&outMain)
			outMain.Free()

			suite.Equal(out, outMain)
		}
	}
}

func testPrecomputePoints(suite *suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor
	cfg.ArePointsSharedInBatch = false

	for _, power := range []int{7, 8} {
		for _, batchSize := range []int{1, 3, 5} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_381.GenerateScalars(totalSize)
			points := icicleBls12_381.GenerateAffinePoints(totalSize)
			cfg.BatchSize = int32(batchSize)

			test_helpers.ActivateReferenceDevice()
			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			suite.Equal(runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			e = msm.PrecomputeBases(points, &cfg, precomputeOut)
			suite.Equal(runtime.Success, e, "PrecomputeBases failed")

			var p icicleBls12_381.Projective
			var out core.DeviceSlice
			_, e = out.Malloc(p.Size(), batchSize)
			suite.Equal(runtime.Success, e, "Allocating bytes on device for Projective results failed")

			e = msm.Msm(scalars, precomputeOut, &cfg, out)
			suite.Equal(runtime.Success, e, "Msm failed")
			outHost := make(core.HostSlice[icicleBls12_381.Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()

			// Main device run
			test_helpers.ActivateMainDevice()
			var precomputeOutMain core.DeviceSlice
			_, eMain := precomputeOutMain.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			suite.Equal(runtime.Success, eMain, "Allocating bytes on device for PrecomputeBases results failed")

			eMain = msm.PrecomputeBases(points, &cfg, precomputeOutMain)
			suite.Equal(runtime.Success, eMain, "PrecomputeBases failed")

			var pMain icicleBls12_381.Projective
			var outMain core.DeviceSlice
			_, eMain = outMain.Malloc(pMain.Size(), batchSize)
			suite.Equal(runtime.Success, eMain, "Allocating bytes on device for Projective results failed")

			eMain = msm.Msm(scalars, precomputeOutMain, &cfg, outMain)
			suite.Equal(runtime.Success, eMain, "Msm failed")
			outHostMain := make(core.HostSlice[icicleBls12_381.Projective], batchSize)
			outHostMain.CopyFromDevice(&outMain)
			outMain.Free()
			precomputeOutMain.Free()

			suite.Equal(out, outMain, "MSM Batch with precompute failed")
		}
	}
}

func testPrecomputePointsSharedBases(suite *suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor

	for _, power := range []int{4, 5, 6} {
		for _, batchSize := range []int{1, 3, 5} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_381.GenerateScalars(totalSize)
			points := icicleBls12_381.GenerateAffinePoints(size)

			test_helpers.ActivateReferenceDevice()
			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			suite.Equal(runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			e = msm.PrecomputeBases(points, &cfg, precomputeOut)
			suite.Equal(runtime.Success, e, "PrecomputeBases failed")

			var p icicleBls12_381.Projective
			var out core.DeviceSlice
			_, e = out.Malloc(p.Size(), batchSize)
			suite.Equal(runtime.Success, e, "Allocating bytes on device for Projective results failed")

			e = msm.Msm(scalars, precomputeOut, &cfg, out)
			suite.Equal(runtime.Success, e, "Msm failed")
			outHost := make(core.HostSlice[icicleBls12_381.Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()

			// Activate Main device
			test_helpers.ActivateMainDevice()
			var precomputeOutMain core.DeviceSlice
			_, eMain := precomputeOutMain.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			suite.Equal(runtime.Success, eMain, "Allocating bytes on device for PrecomputeBases results failed")

			eMain = msm.PrecomputeBases(points, &cfg, precomputeOutMain)
			suite.Equal(runtime.Success, eMain, "PrecomputeBases failed")

			var pMain icicleBls12_381.Projective
			var outMain core.DeviceSlice
			_, eMain = outMain.Malloc(pMain.Size(), batchSize)
			suite.Equal(runtime.Success, eMain, "Allocating bytes on device for Projective results failed")

			eMain = msm.Msm(scalars, precomputeOutMain, &cfg, outMain)
			suite.Equal(runtime.Success, eMain, "Msm failed")
			outHostMain := make(core.HostSlice[icicleBls12_381.Projective], batchSize)
			outHostMain.CopyFromDevice(&outMain)
			outMain.Free()
			precomputeOutMain.Free()

			suite.Equal(out, outMain, "MSM Batch with shared precompute failed")
		}
	}
}

func testMSMSkewedDistribution(suite *suite.Suite) {
	cfg := msm.GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5} {
		size := 1 << power
		scalars := icicleBls12_381.GenerateScalars(size)
		for i := size / 4; i < size; i++ {
			scalars[i].One()
		}
		points := icicleBls12_381.GenerateAffinePoints(size)
		for i := 0; i < size/4; i++ {
			points[i].Zero()
		}

		test_helpers.ActivateReferenceDevice()
		var p icicleBls12_381.Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), 1)
		suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")

		e = msm.Msm(scalars, points, &cfg, out)
		suite.Equal(e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[icicleBls12_381.Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()

		// Main
		test_helpers.ActivateMainDevice()
		var pMain icicleBls12_381.Projective
		var outMain core.DeviceSlice
		_, eMain := outMain.Malloc(pMain.Size(), 1)
		suite.Equal(eMain, runtime.Success, "Allocating bytes on device for Projective results failed")

		eMain = msm.Msm(scalars, points, &cfg, outMain)
		suite.Equal(eMain, runtime.Success, "Msm failed")
		outHostMain := make(core.HostSlice[icicleBls12_381.Projective], 1)
		outHostMain.CopyFromDevice(&outMain)
		outMain.Free()

		suite.Equal(out, outMain, "MSM skewed distribution failed")
	}
}

// TODO - RunOnDevice causes incorrect values
// TODO - Support point and field arithmetics outside of vecops
//func testMSMMultiDevice(suite *suite.Suite) {
//	test_helpers.ActivateMainDevice()
//	secondHalfDevice := runtime.CreateDevice("CUDA", 1)
//	numDevices, _ := runtime.GetDeviceCount()
//	if numDevices < 2 {
//		secondHalfDevice.Id = 0
//	}
//
//	cfg := msm.GetDefaultMSMConfig()
//
//	size := 1 << 10
//	halfSize := size / 2
//
//	scalars := icicleBls12_381.GenerateScalars(size)
//	points := icicleBls12_381.GenerateAffinePoints(size)
//
//	// CPU run
//	test_helpers.ActivateReferenceDevice()
//	outHost := make(core.HostSlice[icicleBls12_381.Projective], 1)
//	e := msm.Msm(core.HostSliceFromElements(scalars), core.HostSliceFromElements(points), &cfg, outHost)
//	suite.Equal(e, runtime.Success, "Msm failed")
//
//	wg := sync.WaitGroup{}
//	wg.Add(2)
//
//	outHostMain1 := make(core.HostSlice[icicleBls12_381.Projective], 1)
//	outHostMain2 := make(core.HostSlice[icicleBls12_381.Projective], 1)
//	// Cuda run
//	runtime.RunOnDevice(&test_helpers.MAIN_DEVICE, func(args ...any) {
//		e = msm.Msm(
//			scalars[:halfSize],
//			points[:halfSize],
//			&cfg,
//			outHostMain1,
//		)
//		suite.Equal(e, runtime.Success, "Msm failed")
//		wg.Done()
//	})
//
//	runtime.RunOnDevice(&secondHalfDevice, func(args ...any) {
//		e = msm.Msm(
//			scalars[halfSize:],
//			points[halfSize:],
//			&cfg,
//			outHostMain2,
//		)
//		suite.Equal(e, runtime.Success, "Msm failed")
//		wg.Done()
//	})
//
//	wg.Wait()
//
//	outHostMain := make(core.HostSlice[icicleBls12_381.Projective], 1)
//	var one icicleBls12_381.ScalarField
//	one.One()
//	ones := []icicleBls12_381.ScalarField{one, one}
//	e = msm.Msm(
//		core.HostSliceFromElements(ones),
//		append(outHostMain1, outHostMain2[0]),
//		&cfg,
//		outHostMain,
//	)
//
//	suite.Equal(outHost, outHostMain)
//}

type MSMTestSuite struct {
	suite.Suite
}

func (s *MSMTestSuite) TestMSM() {
	s.Run("TestMSM", testWrapper(&s.Suite, testMSM))
	s.Run("TestMSMBatch", testWrapper(&s.Suite, testMSMBatch))
	s.Run("TestPrecomputePoints", testWrapper(&s.Suite, testPrecomputePoints))
	s.Run("TestPrecomputePointsSharedBases", testWrapper(&s.Suite, testPrecomputePointsSharedBases))
	s.Run("TestMSMSkewedDistribution", testWrapper(&s.Suite, testMSMSkewedDistribution))
	// s.Run("TestMSMMultiDevice", testWrapper(&s.Suite, testMSMMultiDevice))
}

func TestSuiteMSM(t *testing.T) {
	suite.Run(t, new(MSMTestSuite))
}
