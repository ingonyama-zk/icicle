package tests

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/consensys/gnark-crypto/ecc"
	bls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	"github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	icicleBls12_381 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12381"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12381/g2"
)

func projectiveToGnarkAffineG2(p g2.G2Projective) bls12381.G2Affine {
	pxBytes := p.X.ToBytesLittleEndian()
	pxA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pxBytes[:fp.Bytes]))
	pxA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pxBytes[fp.Bytes:]))
	x := bls12381.E2{
		A0: pxA0,
		A1: pxA1,
	}

	pyBytes := p.Y.ToBytesLittleEndian()
	pyA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pyBytes[:fp.Bytes]))
	pyA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pyBytes[fp.Bytes:]))
	y := bls12381.E2{
		A0: pyA0,
		A1: pyA1,
	}

	pzBytes := p.Z.ToBytesLittleEndian()
	pzA0, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pzBytes[:fp.Bytes]))
	pzA1, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)(pzBytes[fp.Bytes:]))
	z := bls12381.E2{
		A0: pzA0,
		A1: pzA1,
	}

	var zSquared bls12381.E2
	zSquared.Mul(&z, &z)

	var X bls12381.E2
	X.Mul(&x, &z)

	var Y bls12381.E2
	Y.Mul(&y, &zSquared)

	g2Jac := bls12381.G2Jac{
		X: X,
		Y: Y,
		Z: z,
	}

	var g2Affine bls12381.G2Affine
	return *g2Affine.FromJacobian(&g2Jac)
}

func testAgainstGnarkCryptoMsmG2(scalars core.HostSlice[icicleBls12_381.ScalarField], points core.HostSlice[g2.G2Affine], out g2.G2Projective) bool {
	scalarsFr := make([]fr.Element, len(scalars))
	for i, v := range scalars {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		scalarsFr[i] = slice64
	}

	pointsFp := make([]bls12381.G2Affine, len(points))
	for i, v := range points {
		pointsFp[i] = projectiveToGnarkAffineG2(v.ToProjective())
	}

	return testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(scalarsFr, pointsFp, out)
}

func testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(scalarsFr core.HostSlice[fr.Element], pointsFp core.HostSlice[bls12381.G2Affine], out g2.G2Projective) bool {
	var msmRes bls12381.G2Jac
	msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

	var icicleResAsJac bls12381.G2Jac
	proj := projectiveToGnarkAffineG2(out)
	icicleResAsJac.FromAffine(&proj)

	return msmRes.Equal(&icicleResAsJac)
}

func convertIcicleG2AffineToG2Affine(iciclePoints []g2.G2Affine) []bls12381.G2Affine {
	points := make([]bls12381.G2Affine, len(iciclePoints))
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

		points[index] = bls12381.G2Affine{
			X: bls12381.E2{
				A0: xA0Elem,
				A1: xA1Elem,
			},
			Y: bls12381.E2{
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
		size := 1 << power

		scalars := icicleBls12_381.GenerateScalars(size)
		points := g2.G2GenerateAffinePoints(size)

		stream, _ := cr.CreateStream()
		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), p.Size(), stream)
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
		cfg.Ctx.Stream = &stream

		e = g2.G2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		cr.SynchronizeStream(&stream)
		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsmG2(scalars, points, outHost[0]))

	}
}
func TestMSMG2GnarkCryptoTypes(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	for _, power := range []int{3} {
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
		pointsHost := (core.HostSlice[bls12381.G2Affine])(pointsGnark)

		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
		cfg.ArePointsMontgomeryForm = true
		cfg.AreScalarsMontgomeryForm = true

		e = g2.G2Msm(scalarsHost, pointsHost, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()

		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(scalarsHost, pointsHost, outHost[0]))
	}
}

func TestMSMG2Batch(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_381.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(totalSize)

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e := out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			e = g2.G2Msm(scalars, points, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[g2.G2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				assert.True(t, testAgainstGnarkCryptoMsmG2(scalarsSlice, pointsSlice, out))
			}
		}
	}
}

func TestPrecomputeBaseG2(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	const precomputeFactor = 8
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_381.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(totalSize)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for PrecomputeBases results failed")

			e = g2.G2PrecomputeBases(points, precomputeFactor, 0, &cfg.Ctx, precomputeOut)
			assert.Equal(t, e, cr.CudaSuccess, "PrecomputeBases failed")

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			cfg.PrecomputeFactor = precomputeFactor

			e = g2.G2Msm(scalars, precomputeOut, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[g2.G2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				assert.True(t, testAgainstGnarkCryptoMsmG2(scalarsSlice, pointsSlice, out))
			}
		}
	}
}

func TestMSMG2SkewedDistribution(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := icicleBls12_381.GenerateScalars(size)
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
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

		e = g2.G2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()
		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsmG2(scalars, points, outHost[0]))
	}
}

func TestMSMG2MultiDevice(t *testing.T) {
	numDevices, _ := cr.GetDeviceCount()
	numDevices = 1 // TODO remove when test env is fixed
	fmt.Println("There are ", numDevices, " devices available")
	orig_device, _ := cr.GetDevice()
	wg := sync.WaitGroup{}

	for i := 0; i < numDevices; i++ {
		wg.Add(1)
		cr.RunOnDevice(i, func(args ...any) {
			defer wg.Done()
			cfg := g2.G2GetDefaultMSMConfig()
			cfg.IsAsync = true
			for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
				size := 1 << power
				scalars := icicleBls12_381.GenerateScalars(size)
				points := g2.G2GenerateAffinePoints(size)

				stream, _ := cr.CreateStream()
				var p g2.G2Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), p.Size(), stream)
				assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
				cfg.Ctx.Stream = &stream

				e = g2.G2Msm(scalars, points, &cfg, out)
				assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
				outHost := make(core.HostSlice[g2.G2Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				cr.SynchronizeStream(&stream)
				// Check with gnark-crypto
				assert.True(t, testAgainstGnarkCryptoMsmG2(scalars, points, outHost[0]))
			}
		})
	}
	wg.Wait()
	cr.SetDevice(orig_device)
}
