package g2

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bls12-377"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fp"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fr"
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	icicleBls12_377 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bls12377"
)

func projectiveToGnarkAffine(p icicleBls12_377.Projective) bls12377.G1Affine {
	px, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)((&p.X).ToBytesLittleEndian()))
	py, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)((&p.Y).ToBytesLittleEndian()))
	pz, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)((&p.Z).ToBytesLittleEndian()))

	zInv := new(fp.Element)
	x := new(fp.Element)
	y := new(fp.Element)

	zInv.Inverse(&pz)

	x.Mul(&px, zInv)
	y.Mul(&py, zInv)

	return bls12377.G1Affine{X: *x, Y: *y}
}

func testAgainstGnarkCryptoMsm_g2(scalars core.HostSlice[icicleBls12_377.ScalarField], points core.HostSlice[icicleBls12_377._g2Affine], out icicleBls12_377._g2Projective) bool {
	scalarsFr := make([]fr.Element, len(scalars))
	for i, v := range scalars {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		scalarsFr[i] = slice64
	}

	pointsFp := make([]bls12377.G1Affine, len(points))
	for i, v := range points {
		pointsFp[i] = projectiveToGnarkAffine_g2(v.ToProjective())
	}

	return testAgainstGnarkCryptoMsm_g2GnarkCryptoTypes(scalarsFr, pointsFp, out)
}

func testAgainstGnarkCryptoMsm_g2GnarkCryptoTypes(scalarsFr core.HostSlice[fr.Element], pointsFp core.HostSlice[bls12377.G1Affine], out icicleBls12_377._g2Projective) bool {
	var msmRes bls12377.G1Jac
	msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

	var icicleResAsJac bls12377.G1Jac
	proj := projectiveToGnarkAffine_g2(out)
	icicleResAsJac.FromAffine(&proj)

	return msmRes.Equal(&icicleResAsJac)
}

func convertIcicleAffineToG1Affine(iciclePoints []icicleBls12_377._g2Affine) []bls12377.G1Affine {
	points := make([]bls12377.G1Affine, len(iciclePoints))
	for index, iciclePoint := range iciclePoints {
		xBytes := ([fp.Bytes]byte)(iciclePoint.X.ToBytesLittleEndian())
		fpXElem, _ := fp.LittleEndian.Element(&xBytes)

		yBytes := ([fp.Bytes]byte)(iciclePoint.Y.ToBytesLittleEndian())
		fpYElem, _ := fp.LittleEndian.Element(&yBytes)
		points[index] = bls12377.G1Affine{
			X: fpXElem,
			Y: fpYElem,
		}
	}

	return points
}

func TestMSM_g2(t *testing.T) {
	cfg := _g2GetDefaultMSMConfig()
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := icicleBls12_377.GenerateScalars(size)
		points := icicleBls12_377._g2GenerateAffinePoints(size)

		stream, _ := cr.CreateStream()
		var p icicleBls12_377._g2Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), p.Size(), stream)
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
		cfg.Ctx.Stream = &stream

		e = _g2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[icicleBls12_377._g2Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		cr.SynchronizeStream(&stream)
		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsm_g2(scalars, points, outHost[0]))

	}
}
func TestMSMGnarkCryptoTypes(t *testing.T) {
	cfg := _g2GetDefaultMSMConfig()
	for _, power := range []int{3} {
		size := 1 << power

		scalars := make([]fr.Element, size)
		var x fr.Element
		for i := 0; i < size; i++ {
			x.SetRandom()
			scalars[i] = x
		}
		scalarsHost := (core.HostSlice[fr.Element])(scalars)
		points := icicleBls12_377._g2GenerateAffinePoints(size)
		pointsGnark := convertIcicleAffineToG1Affine(points)
		pointsHost := (core.HostSlice[bls12377.G1Affine])(pointsGnark)

		var p icicleBls12_377._g2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
		cfg.ArePointsMontgomeryForm = true
		cfg.AreScalarsMontgomeryForm = true

		e = _g2Msm(scalarsHost, pointsHost, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[icicleBls12_377._g2Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()

		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsm_g2GnarkCryptoTypes(scalarsHost, pointsHost, outHost[0]))
	}
}

func TestMSM_g2Batch(t *testing.T) {
	cfg := _g2GetDefaultMSMConfig()
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_377.GenerateScalars(totalSize)
			points := icicleBls12_377._g2GenerateAffinePoints(totalSize)

			var p icicleBls12_377._g2Projective
			var out core.DeviceSlice
			_, e := out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			e = _g2Msm(scalars, points, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[icicleBls12_377._g2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				assert.True(t, testAgainstGnarkCryptoMsm_g2(scalarsSlice, pointsSlice, out))
			}
		}
	}
}

func TestPrecomputeBase_g2(t *testing.T) {
	cfg := _g2GetDefaultMSMConfig()
	const precomputeFactor = 8
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBls12_377.GenerateScalars(totalSize)
			points := icicleBls12_377._g2GenerateAffinePoints(totalSize)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for PrecomputeBases results failed")

			e = _g2PrecomputeBases(points, precomputeFactor, 0, &cfg.Ctx, precomputeOut)
			assert.Equal(t, e, cr.CudaSuccess, "PrecomputeBases failed")

			var p icicleBls12_377._g2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			cfg.PrecomputeFactor = precomputeFactor

			e = _g2Msm(scalars, precomputeOut, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[icicleBls12_377._g2Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()
			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				assert.True(t, testAgainstGnarkCryptoMsm_g2(scalarsSlice, pointsSlice, out))
			}
		}
	}
}

func TestMSM_g2SkewedDistribution(t *testing.T) {
	cfg := _g2GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := icicleBls12_377.GenerateScalars(size)
		for i := size / 4; i < size; i++ {
			scalars[i].One()
		}
		points := icicleBls12_377._g2GenerateAffinePoints(size)
		for i := 0; i < size/4; i++ {
			points[i].Zero()
		}

		var p icicleBls12_377._g2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

		e = _g2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[icicleBls12_377._g2Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()
		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsm_g2(scalars, points, outHost[0]))
	}
}

func TestMSM_g2MultiDevice(t *testing.T) {
	numDevices, _ := cr.GetDeviceCount()
	numDevices = 1 // TODO remove when test env is fixed
	fmt.Println("There are ", numDevices, " devices available")
	orig_device, _ := cr.GetDevice()
	wg := sync.WaitGroup{}

	for i := 0; i < numDevices; i++ {
		wg.Add(1)
		cr.RunOnDevice(i, func(args ...any) {
			defer wg.Done()
			cfg := _g2GetDefaultMSMConfig()
			cfg.IsAsync = true
			for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
				size := 1 << power
				scalars := icicleBls12_377.GenerateScalars(size)
				points := icicleBls12_377._g2GenerateAffinePoints(size)

				stream, _ := cr.CreateStream()
				var p icicleBls12_377._g2Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), p.Size(), stream)
				assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
				cfg.Ctx.Stream = &stream

				e = _g2Msm(scalars, points, &cfg, out)
				assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
				outHost := make(core.HostSlice[icicleBls12_377._g2Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				cr.SynchronizeStream(&stream)
				// Check with gnark-crypto
				assert.True(t, testAgainstGnarkCryptoMsm_g2(scalars, points, outHost[0]))
			}
		})
	}
	wg.Wait()
	cr.SetDevice(orig_device)
}
