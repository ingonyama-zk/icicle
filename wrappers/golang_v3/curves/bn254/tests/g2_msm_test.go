package tests

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	icicleBn254 "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bn254"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bn254/g2"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
)

func projectiveToGnarkAffineG2(p g2.G2Projective) bn254.G2Affine {
	pxBytes := p.X.ToBytesLittleEndian()
	pxA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pxBytes[:fp.Bytes]))
	pxA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pxBytes[fp.Bytes:]))
	x := bn254.E2{
		A0: pxA0,
		A1: pxA1,
	}

	pyBytes := p.Y.ToBytesLittleEndian()
	pyA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pyBytes[:fp.Bytes]))
	pyA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pyBytes[fp.Bytes:]))
	y := bn254.E2{
		A0: pyA0,
		A1: pyA1,
	}

	pzBytes := p.Z.ToBytesLittleEndian()
	pzA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pzBytes[:fp.Bytes]))
	pzA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pzBytes[fp.Bytes:]))
	z := bn254.E2{
		A0: pzA0,
		A1: pzA1,
	}

	var zSquared bn254.E2
	zSquared.Mul(&z, &z)

	var X bn254.E2
	X.Mul(&x, &z)

	var Y bn254.E2
	Y.Mul(&y, &zSquared)

	g2Jac := bn254.G2Jac{
		X: X,
		Y: Y,
		Z: z,
	}

	var g2Affine bn254.G2Affine
	return *g2Affine.FromJacobian(&g2Jac)
}

func testAgainstGnarkCryptoMsmG2(t *testing.T, scalars core.HostSlice[icicleBn254.ScalarField], points core.HostSlice[g2.G2Affine], out g2.G2Projective) {
	scalarsFr := make([]fr.Element, len(scalars))
	for i, v := range scalars {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		scalarsFr[i] = slice64
	}

	pointsFp := make([]bn254.G2Affine, len(points))
	for i, v := range points {
		pointsFp[i] = projectiveToGnarkAffineG2(v.ToProjective())
	}

	testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(t, scalarsFr, pointsFp, out)
}

func testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(t *testing.T, scalarsFr core.HostSlice[fr.Element], pointsFp core.HostSlice[bn254.G2Affine], out g2.G2Projective) {
	var msmRes bn254.G2Jac
	msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

	var msmResAffine bn254.G2Affine
	msmResAffine.FromJacobian(&msmRes)

	icicleResAffine := projectiveToGnarkAffineG2(out)

	assert.Equal(t, msmResAffine, icicleResAffine)
}

func convertIcicleG2AffineToG2Affine(iciclePoints []g2.G2Affine) []bn254.G2Affine {
	points := make([]bn254.G2Affine, len(iciclePoints))
	for index, iciclePoint := range iciclePoints {
		xBytes := ([fp.Bytes * 2]byte)(iciclePoint.X.ToBytesLittleEndian())
		xA0Bytes := ([fp.Bytes]byte)(xBytes[:fp.Bytes])
		xA1Bytes := ([fp.Bytes]byte)(xBytes[fp.Bytes:])
		xA0Elem, _ := fp.LittleEndian.Element(&xA0Bytes)
		xA1Elem, _ := fp.LittleEndian.Element(&xA1Bytes)

		yBytes := ([fp.Bytes * 2]byte)(iciclePoint.Y.ToBytesLittleEndian())
		yA0Bytes := ([fp.Bytes]byte)(yBytes[:fp.Bytes])
		yA1Bytes := ([fp.Bytes]byte)(yBytes[fp.Bytes:])
		yA0Elem, _ := fp.LittleEndian.Element(&yA0Bytes)
		yA1Elem, _ := fp.LittleEndian.Element(&yA1Bytes)

		points[index] = bn254.G2Affine{
			X: bn254.E2{
				A0: xA0Elem,
				A1: xA1Elem,
			},
			Y: bn254.E2{
				A0: yA0Elem,
				A1: yA1Elem,
			},
		}
	}

	return points
}

func TestMSMG2(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		runtime.SetDevice(&DEVICE)
		size := 1 << power

		scalars := icicleBn254.GenerateScalars(size)
		points := g2.G2GenerateAffinePoints(size)

		stream, _ := runtime.CreateStream()
		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), p.Size(), stream)
		assert.Equal(t, e, runtime.Success, "Allocating bytes on device for Projective results failed")
		cfg.StreamHandle = stream

		e = g2.G2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		runtime.SynchronizeStream(stream)
		// Check with gnark-crypto
		testAgainstGnarkCryptoMsmG2(t, scalars, points, outHost[0])
	}
}

//	func TestMSMG2PinnedHostMemory(t *testing.T) {
//		cfg := g2.G2GetDefaultMSMConfig()
//		for _, power := range []int{10} {
//			size := 1 << power
//
//			scalars := icicleBn254.GenerateScalars(size)
//			points := g2.G2GenerateAffinePoints(size)
//
//			pinnable := cr.GetDeviceAttribute(cr.CudaDevAttrHostRegisterSupported, 0)
//			lockable := cr.GetDeviceAttribute(cr.CudaDevAttrPageableMemoryAccessUsesHostPageTables, 0)
//
//			pinnableAndLockable := pinnable == 1 && lockable == 0
//
//			var pinnedPoints core.HostSlice[g2.G2Affine]
//			if pinnableAndLockable {
//				points.Pin(cr.CudaHostRegisterDefault)
//				pinnedPoints, _ = points.AllocPinned(cr.CudaHostAllocDefault)
//				assert.Equal(t, points, pinnedPoints, "Allocating newly pinned memory resulted in bad points")
//			}
//
//			var p g2.G2Projective
//			var out core.DeviceSlice
//			_, e := out.Malloc(p.Size(), p.Size())
//			assert.Equal(t, e, runtime.Success, "Allocating bytes on device for Projective results failed")
//			outHost := make(core.HostSlice[g2.G2Projective], 1)
//
//			e = g2.G2Msm(scalars, points, &cfg, out)
//			assert.Equal(t, e, runtime.Success, "Msm allocated pinned host mem failed")
//
//			outHost.CopyFromDevice(&out)
//			// 		// Check with gnark-crypto
//			assert.True(t, testAgainstGnarkCryptoMsmG2(scalars, points, outHost[0]))
//
//
//			if pinnableAndLockable {
//			e = g2.G2Msm(scalars, pinnedPoints, &cfg, out)
//				assert.Equal(t, e, runtime.Success, "Msm registered pinned host mem failed")
//
//				outHost.CopyFromDevice(&out)
//				// 			// Check with gnark-crypto
//				assert.True(t, testAgainstGnarkCryptoMsmG2(scalars, pinnedPoints, outHost[0]))
//
//			}
//
//			out.Free()
//
//			if pinnableAndLockable {
//				points.Unpin()
//				pinnedPoints.FreePinned()
//			}
//		}
//	}
func TestMSMG2GnarkCryptoTypes(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	for _, power := range []int{3} {
		runtime.SetDevice(&DEVICE)
		size := 1 << power

		scalars := make([]fr.Element, size)
		var x fr.Element
		for i := 0; i < size; i++ {
			x.SetRandom()
			scalars[i] = x
		}
		scalarsHost := (core.HostSlice[fr.Element])(scalars)
		points := g2.G2GenerateAffinePoints(size)
		pointsGnark := convertIcicleG2AffineToG2Affine(points)
		pointsHost := (core.HostSlice[bn254.G2Affine])(pointsGnark)

		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, runtime.Success, "Allocating bytes on device for Projective results failed")
		cfg.AreBasesMontgomeryForm = true
		cfg.AreScalarsMontgomeryForm = true

		e = g2.G2Msm(scalarsHost, pointsHost, &cfg, out)
		assert.Equal(t, e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()

		// Check with gnark-crypto
		testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(t, scalarsHost, pointsHost, outHost[0])
	}
}

func TestMSMG2Batch(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			runtime.SetDevice(&DEVICE)
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBn254.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(totalSize)

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e := out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, runtime.Success, "Allocating bytes on device for Projective results failed")

			e = g2.G2Msm(scalars, points, &cfg, out)
			assert.Equal(t, e, runtime.Success, "Msm failed")
			outHost := make(core.HostSlice[g2.G2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				testAgainstGnarkCryptoMsmG2(t, scalarsSlice, pointsSlice, out)
			}
		}
	}
}

func TestPrecomputePointsG2(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor

	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			runtime.SetDevice(&DEVICE)

			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBn254.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(totalSize)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())
			assert.Equal(t, runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			cfg.BatchSize = int32(batchSize)
			cfg.AreBasesShared = false
			e = g2.G2PrecomputeBases(points, &cfg, precomputeOut)
			assert.Equal(t, runtime.Success, e, "PrecomputeBases failed")

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, runtime.Success, e, "Allocating bytes on device for Projective results failed")

			e = g2.G2Msm(scalars, precomputeOut, &cfg, out)
			assert.Equal(t, runtime.Success, e, "Msm failed")
			outHost := make(core.HostSlice[g2.G2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				testAgainstGnarkCryptoMsmG2(t, scalarsSlice, pointsSlice, out)
			}
		}
	}
}

func TestPrecomputePointsSharedBasesG2(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor

	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			runtime.SetDevice(&DEVICE)

			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBn254.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(size)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())
			assert.Equal(t, runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			e = g2.G2PrecomputeBases(points, &cfg, precomputeOut)
			assert.Equal(t, runtime.Success, e, "PrecomputeBases failed")

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, runtime.Success, e, "Allocating bytes on device for Projective results failed")

			e = g2.G2Msm(scalars, precomputeOut, &cfg, out)
			assert.Equal(t, runtime.Success, e, "Msm failed")
			outHost := make(core.HostSlice[g2.G2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[0:size]
				out := outHost[i]
				testAgainstGnarkCryptoMsmG2(t, scalarsSlice, pointsSlice, out)
			}
		}
	}
}

func TestMSMG2SkewedDistribution(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		runtime.SetDevice(&DEVICE)

		size := 1 << power

		scalars := icicleBn254.GenerateScalars(size)
		for i := size / 4; i < size; i++ {
			scalars[i].One()
		}
		points := g2.G2GenerateAffinePoints(size)
		for i := 0; i < size/4; i++ {
			points[i].Zero()
		}

		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, runtime.Success, "Allocating bytes on device for Projective results failed")

		e = g2.G2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()
		// Check with gnark-crypto
		testAgainstGnarkCryptoMsmG2(t, scalars, points, outHost[0])
	}
}

func TestMSMG2MultiDevice(t *testing.T) {
	numDevices, _ := runtime.GetDeviceCount()
	fmt.Println("There are ", numDevices, " ", DEVICE.GetDeviceType(), " devices available")
	wg := sync.WaitGroup{}

	for i := 0; i < numDevices; i++ {
		currentDevice := runtime.Device{DeviceType: DEVICE.DeviceType, Id: int32(i)}
		wg.Add(1)
		runtime.RunOnDevice(&currentDevice, func(args ...any) {
			defer wg.Done()

			fmt.Println("Running on ", currentDevice.GetDeviceType(), " ", currentDevice.Id, " device")

			cfg := g2.G2GetDefaultMSMConfig()
			cfg.IsAsync = true
			for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
				size := 1 << power
				scalars := icicleBn254.GenerateScalars(size)
				points := g2.G2GenerateAffinePoints(size)

				stream, _ := runtime.CreateStream()
				var p g2.G2Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), p.Size(), stream)
				assert.Equal(t, e, runtime.Success, "Allocating bytes on device for Projective results failed")
				cfg.StreamHandle = stream

				e = g2.G2Msm(scalars, points, &cfg, out)
				assert.Equal(t, e, runtime.Success, "Msm failed")
				outHost := make(core.HostSlice[g2.G2Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				runtime.SynchronizeStream(stream)
				// Check with gnark-crypto
				testAgainstGnarkCryptoMsmG2(t, scalars, points, outHost[0])
			}
		})
	}
	wg.Wait()
}
