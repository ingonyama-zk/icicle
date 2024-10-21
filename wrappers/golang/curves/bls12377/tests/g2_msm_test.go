package tests

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bls12-377"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fp"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fr"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	icicleBls12_377 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/g2"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func projectiveToGnarkAffineG2(p g2.G2Projective) bls12377.G2Affine {
	pxBytes := p.X.ToBytesLittleEndian()
	pxA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pxBytes[:fp.Bytes]))
	pxA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pxBytes[fp.Bytes:]))
	x := bls12377.E2{
		A0: pxA0,
		A1: pxA1,
	}

	pyBytes := p.Y.ToBytesLittleEndian()
	pyA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pyBytes[:fp.Bytes]))
	pyA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pyBytes[fp.Bytes:]))
	y := bls12377.E2{
		A0: pyA0,
		A1: pyA1,
	}

	pzBytes := p.Z.ToBytesLittleEndian()
	pzA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pzBytes[:fp.Bytes]))
	pzA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pzBytes[fp.Bytes:]))
	z := bls12377.E2{
		A0: pzA0,
		A1: pzA1,
	}

	var zSquared bls12377.E2
	zSquared.Mul(&z, &z)

	var X bls12377.E2
	X.Mul(&x, &z)

	var Y bls12377.E2
	Y.Mul(&y, &zSquared)

	g2Jac := bls12377.G2Jac{
		X: X,
		Y: Y,
		Z: z,
	}

	var g2Affine bls12377.G2Affine
	return *g2Affine.FromJacobian(&g2Jac)
}

func testAgainstGnarkCryptoMsmG2(suite suite.Suite, scalars core.HostSlice[icicleBls12_377.ScalarField], points core.HostSlice[g2.G2Affine], out g2.G2Projective) {
	scalarsFr := make([]fr.Element, len(scalars))
	for i, v := range scalars {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		scalarsFr[i] = slice64
	}

	pointsFp := make([]bls12377.G2Affine, len(points))
	for i, v := range points {
		pointsFp[i] = projectiveToGnarkAffineG2(v.ToProjective())
	}

	testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(suite, scalarsFr, pointsFp, out)
}

func testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(suite suite.Suite, scalarsFr core.HostSlice[fr.Element], pointsFp core.HostSlice[bls12377.G2Affine], out g2.G2Projective) {
	var msmRes bls12377.G2Jac
	msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

	var msmResAffine bls12377.G2Affine
	msmResAffine.FromJacobian(&msmRes)

	icicleResAffine := projectiveToGnarkAffineG2(out)

	suite.Equal(msmResAffine, icicleResAffine)
}

func convertIcicleG2AffineToG2Affine(iciclePoints []g2.G2Affine) []bls12377.G2Affine {
	points := make([]bls12377.G2Affine, len(iciclePoints))
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

		points[index] = bls12377.G2Affine{
			X: bls12377.E2{
				A0: xA0Elem,
				A1: xA1Elem,
			},
			Y: bls12377.E2{
				A0: yA0Elem,
				A1: yA1Elem,
			},
		}
	}

	return points
}

func testMSMG2(suite suite.Suite) {
	cfg := g2.G2GetDefaultMSMConfig()
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6} {
		runtime.SetDevice(&DEVICE)
		size := 1 << power

		scalars := icicleBls12_377.GenerateScalars(size)
		points := g2.G2GenerateAffinePoints(size)

		stream, _ := runtime.CreateStream()
		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), 1, stream)
		suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")
		cfg.StreamHandle = stream

		e = g2.G2Msm(scalars, points, &cfg, out)
		suite.Equal(e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		runtime.SynchronizeStream(stream)
		runtime.DestroyStream(stream)
		// Check with gnark-crypto
		testAgainstGnarkCryptoMsmG2(suite, scalars, points, outHost[0])
	}
}

func testMSMG2GnarkCryptoTypes(suite suite.Suite) {
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
		pointsHost := (core.HostSlice[bls12377.G2Affine])(pointsGnark)

		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), 1)
		suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")
		cfg.AreBasesMontgomeryForm = true
		cfg.AreScalarsMontgomeryForm = true

		e = g2.G2Msm(scalarsHost, pointsHost, &cfg, out)
		suite.Equal(e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()

		// Check with gnark-crypto
		testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(suite, scalarsHost, pointsHost, outHost[0])
	}
}

func testMSMG2Batch(suite suite.Suite) {
	cfg := g2.G2GetDefaultMSMConfig()
	for _, power := range []int{5, 6} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_377.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(totalSize)

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e := out.Malloc(p.Size(), batchSize)
			suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")

			e = g2.G2Msm(scalars, points, &cfg, out)
			suite.Equal(e, runtime.Success, "Msm failed")
			outHost := make(core.HostSlice[g2.G2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				testAgainstGnarkCryptoMsmG2(suite, scalarsSlice, pointsSlice, out)
			}
		}
	}
}

func testPrecomputePointsG2(suite suite.Suite) {
	if DEVICE.GetDeviceType() == "CPU" {
		suite.T().Skip("Skipping cpu test")
	}
	cfg := g2.G2GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor

	for _, power := range []int{7, 8} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)

			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_377.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(totalSize)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			suite.Equal(runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			cfg.BatchSize = int32(batchSize)
			cfg.ArePointsSharedInBatch = false
			e = g2.G2PrecomputeBases(points, &cfg, precomputeOut)
			suite.Equal(runtime.Success, e, "PrecomputeBases failed")

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(p.Size(), batchSize)
			suite.Equal(runtime.Success, e, "Allocating bytes on device for Projective results failed")

			e = g2.G2Msm(scalars, precomputeOut, &cfg, out)
			suite.Equal(runtime.Success, e, "Msm failed")
			outHost := make(core.HostSlice[g2.G2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				testAgainstGnarkCryptoMsmG2(suite, scalarsSlice, pointsSlice, out)
			}
		}
	}
}

func testPrecomputePointsSharedBasesG2(suite suite.Suite) {
	cfg := g2.G2GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor

	for _, power := range []int{4, 5, 6} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)

			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_377.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(size)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			suite.Equal(runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			e = g2.G2PrecomputeBases(points, &cfg, precomputeOut)
			suite.Equal(runtime.Success, e, "PrecomputeBases failed")

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(p.Size(), batchSize)
			suite.Equal(runtime.Success, e, "Allocating bytes on device for Projective results failed")

			e = g2.G2Msm(scalars, precomputeOut, &cfg, out)
			suite.Equal(runtime.Success, e, "Msm failed")
			outHost := make(core.HostSlice[g2.G2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[0:size]
				out := outHost[i]
				testAgainstGnarkCryptoMsmG2(suite, scalarsSlice, pointsSlice, out)
			}
		}
	}
}

func testMSMG2SkewedDistribution(suite suite.Suite) {
	cfg := g2.G2GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5} {
		runtime.SetDevice(&DEVICE)

		size := 1 << power

		scalars := icicleBls12_377.GenerateScalars(size)
		for i := size / 4; i < size; i++ {
			scalars[i].One()
		}
		points := g2.G2GenerateAffinePoints(size)
		for i := 0; i < size/4; i++ {
			points[i].Zero()
		}

		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), 1)
		suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")

		e = g2.G2Msm(scalars, points, &cfg, out)
		suite.Equal(e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()
		// Check with gnark-crypto
		testAgainstGnarkCryptoMsmG2(suite, scalars, points, outHost[0])
	}
}

func testMSMG2MultiDevice(suite suite.Suite) {
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
			for _, power := range []int{2, 3, 4, 5, 6} {
				size := 1 << power
				scalars := icicleBls12_377.GenerateScalars(size)
				points := g2.G2GenerateAffinePoints(size)

				stream, _ := runtime.CreateStream()
				var p g2.G2Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), 1, stream)
				suite.Equal(e, runtime.Success, "Allocating bytes on device for Projective results failed")
				cfg.StreamHandle = stream

				e = g2.G2Msm(scalars, points, &cfg, out)
				suite.Equal(e, runtime.Success, "Msm failed")
				outHost := make(core.HostSlice[g2.G2Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				runtime.SynchronizeStream(stream)
				runtime.DestroyStream(stream)
				// Check with gnark-crypto
				testAgainstGnarkCryptoMsmG2(suite, scalars, points, outHost[0])
			}
		})
	}
	wg.Wait()
}

type MSMG2TestSuite struct {
	suite.Suite
}

func (s *MSMG2TestSuite) TestMSMG2() {
	s.Run("TestMSMG2", testWrapper(s.Suite, testMSMG2))
	s.Run("TestMSMG2GnarkCryptoTypes", testWrapper(s.Suite, testMSMG2GnarkCryptoTypes))
	s.Run("TestMSMG2Batch", testWrapper(s.Suite, testMSMG2Batch))
	s.Run("TestPrecomputePointsG2", testWrapper(s.Suite, testPrecomputePointsG2))
	s.Run("TestPrecomputePointsSharedBasesG2", testWrapper(s.Suite, testPrecomputePointsSharedBasesG2))
	s.Run("TestMSMG2SkewedDistribution", testWrapper(s.Suite, testMSMG2SkewedDistribution))
	s.Run("TestMSMG2MultiDevice", testWrapper(s.Suite, testMSMG2MultiDevice))
}

func TestSuiteMSMG2(t *testing.T) {
	suite.Run(t, new(MSMG2TestSuite))
}
