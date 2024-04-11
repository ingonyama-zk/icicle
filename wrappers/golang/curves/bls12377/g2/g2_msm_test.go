package g2

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	icicle_bls12377 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bls12377"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bls12-377"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fp"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fr"
)

func projectiveToGnarkAffineG2(p G2Projective) bls12377.G2Affine {
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

func testAgainstGnarkCryptoMsmG2(scalars core.HostSlice[icicle_bls12377.ScalarField], points core.HostSlice[G2Affine], out G2Projective) bool {
	scalarsFr := make([]fr.Element, len(scalars))
	for i, v := range scalars {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		scalarsFr[i] = slice64
	}

	pointsFp := make([]bls12377.G2Affine, len(points))
	for i, v := range points {
		pointsFp[i] = projectiveToGnarkAffineG2(v.ToProjective())
	}
	var msmRes bls12377.G2Jac
	msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

	var icicleResAsJac bls12377.G2Jac
	proj := projectiveToGnarkAffineG2(out)
	icicleResAsJac.FromAffine(&proj)

	return msmRes.Equal(&icicleResAsJac)
}

func TestMSMG2(t *testing.T) {
	cfg := icicle_bls12377.GetDefaultMSMConfig()
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := icicle_bls12377.GenerateScalars(size)
		points := G2GenerateAffinePoints(size)

		stream, _ := cr.CreateStream()
		var p G2Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), p.Size(), stream)
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
		cfg.Ctx.Stream = &stream

		e = G2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[G2Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		cr.SynchronizeStream(&stream)
		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsmG2(scalars, points, outHost[0]))
	}
}

func TestMSMG2Batch(t *testing.T) {
	cfg := icicle_bls12377.GetDefaultMSMConfig()
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicle_bls12377.GenerateScalars(totalSize)
			points := G2GenerateAffinePoints(totalSize)

			var p G2Projective
			var out core.DeviceSlice
			_, e := out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			e = G2Msm(scalars, points, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[G2Projective], batchSize)
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
	cfg := icicle_bls12377.GetDefaultMSMConfig()
	const precomputeFactor = 8
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicle_bls12377.GenerateScalars(totalSize)
			points := G2GenerateAffinePoints(totalSize)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for PrecomputeBases results failed")

			e = G2PrecomputeBases(points, precomputeFactor, 0, &cfg.Ctx, precomputeOut)
			assert.Equal(t, e, cr.CudaSuccess, "PrecomputeBases failed")

			var p G2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			cfg.PrecomputeFactor = precomputeFactor

			e = G2Msm(scalars, precomputeOut, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[G2Projective], batchSize)
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
	cfg := icicle_bls12377.GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := icicle_bls12377.GenerateScalars(size)
		for i := size / 4; i < size; i++ {
			scalars[i].One()
		}
		points := G2GenerateAffinePoints(size)
		for i := 0; i < size/4; i++ {
			points[i].Zero()
		}

		var p G2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

		e = G2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[G2Projective], 1)
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
			cfg := icicle_bls12377.GetDefaultMSMConfig()
			cfg.IsAsync = true
			for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
				size := 1 << power
				scalars := icicle_bls12377.GenerateScalars(size)
				points := G2GenerateAffinePoints(size)

				stream, _ := cr.CreateStream()
				var p G2Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), p.Size(), stream)
				assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
				cfg.Ctx.Stream = &stream

				e = G2Msm(scalars, points, &cfg, out)
				assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
				outHost := make(core.HostSlice[G2Projective], 1)
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
