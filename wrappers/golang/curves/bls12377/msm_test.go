package bls12377

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bls12-377"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fp"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fr"
)

func projectiveToGnarkAffine(p Projective) bls12377.G1Affine {
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

func testAgainstGnarkCryptoMsm(scalars core.HostSlice[ScalarField], points core.HostSlice[Affine], out Projective) bool {
	scalarsFr := make([]fr.Element, len(scalars))
	for i, v := range scalars {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		scalarsFr[i] = slice64
	}

	pointsFp := make([]bls12377.G1Affine, len(points))
	for i, v := range points {
		pointsFp[i] = projectiveToGnarkAffine(v.ToProjective())
	}
	var msmRes bls12377.G1Jac
	msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

	var icicleResAsJac bls12377.G1Jac
	proj := projectiveToGnarkAffine(out)
	icicleResAsJac.FromAffine(&proj)

	return msmRes.Equal(&icicleResAsJac)
}

func TestMSM(t *testing.T) {
	cfg := GetDefaultMSMConfig()
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := GenerateScalars(size)
		points := GenerateAffinePoints(size)

		stream, _ := cr.CreateStream()
		var p Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), p.Size(), stream)
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
		cfg.Ctx.Stream = &stream

		e = Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		cr.SynchronizeStream(&stream)
		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsm(scalars, points, outHost[0]))
	}
}

func TestMSMBatch(t *testing.T) {
	cfg := GetDefaultMSMConfig()
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := GenerateScalars(totalSize)
			points := GenerateAffinePoints(totalSize)

			var p Projective
			var out core.DeviceSlice
			_, e := out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			e = Msm(scalars, points, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()

			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				assert.True(t, testAgainstGnarkCryptoMsm(scalarsSlice, pointsSlice, out))
			}
		}
	}
}

func TestPrecomputeBase(t *testing.T) {
	cfg := GetDefaultMSMConfig()
	const precomputeFactor = 8
	for _, power := range []int{10, 16} {
		for _, batchSize := range []int{1, 3, 16} {
			size := 1 << power
			totalSize := size * batchSize
			scalars := GenerateScalars(totalSize)
			points := GenerateAffinePoints(totalSize)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for PrecomputeBases results failed")

			e = PrecomputeBases(points, precomputeFactor, 0, &cfg.Ctx, precomputeOut)
			assert.Equal(t, e, cr.CudaSuccess, "PrecomputeBases failed")

			var p Projective
			var out core.DeviceSlice
			_, e = out.Malloc(batchSize*p.Size(), p.Size())
			assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

			cfg.PrecomputeFactor = precomputeFactor

			e = Msm(scalars, precomputeOut, &cfg, out)
			assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
			outHost := make(core.HostSlice[Projective], batchSize)
			outHost.CopyFromDevice(&out)
			out.Free()
			precomputeOut.Free()

			// Check with gnark-crypto
			for i := 0; i < batchSize; i++ {
				scalarsSlice := scalars[i*size : (i+1)*size]
				pointsSlice := points[i*size : (i+1)*size]
				out := outHost[i]
				assert.True(t, testAgainstGnarkCryptoMsm(scalarsSlice, pointsSlice, out))
			}
		}
	}
}

func TestMSMSkewedDistribution(t *testing.T) {
	cfg := GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := GenerateScalars(size)
		for i := size / 4; i < size; i++ {
			scalars[i].One()
		}
		points := GenerateAffinePoints(size)
		for i := 0; i < size/4; i++ {
			points[i].Zero()
		}

		var p Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

		e = Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := make(core.HostSlice[Projective], 1)
		outHost.CopyFromDevice(&out)
		out.Free()

		// Check with gnark-crypto
		assert.True(t, testAgainstGnarkCryptoMsm(scalars, points, outHost[0]))
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
			cfg := GetDefaultMSMConfig()
			cfg.IsAsync = true
			for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
				size := 1 << power
				scalars := GenerateScalars(size)
				points := GenerateAffinePoints(size)

				stream, _ := cr.CreateStream()
				var p Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), p.Size(), stream)
				assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")
				cfg.Ctx.Stream = &stream

				e = Msm(scalars, points, &cfg, out)
				assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
				outHost := make(core.HostSlice[Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				cr.SynchronizeStream(&stream)
				// Check with gnark-crypto
				assert.True(t, testAgainstGnarkCryptoMsm(scalars, points, outHost[0]))
			}
		})
	}
	wg.Wait()
	cr.SetDevice(orig_device)
}
