package bn254

import (
	"testing"

	"local/hello/icicle/wrappers/golang/core"
	cr "local/hello/icicle/wrappers/golang/cuda_runtime"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/stretchr/testify/assert"
)

func projectiveToGnarkAffine(p Projective) bn254.G1Affine {
	px, _ := fp.LittleEndian.Element((*[32]byte)((&p.X).ToBytesLittleEndian()))
	py, _ := fp.LittleEndian.Element((*[32]byte)((&p.Y).ToBytesLittleEndian()))
	pz, _ := fp.LittleEndian.Element((*[32]byte)((&p.Z).ToBytesLittleEndian()))

	zInv := new(fp.Element)
	x := new(fp.Element)
	y := new(fp.Element)

	zInv.Inverse(&pz)

	x.Mul(&px, zInv)
	y.Mul(&py, zInv)

	return bn254.G1Affine{X: *x, Y: *y}
}

func TestMSM(t *testing.T) {
	cfg := GetDefaultMSMConfig()
	for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
		size := 1 << power

		scalars := GenerateScalars(size)
		points := GenerateAffinePoints(size)

		var p Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), p.Size())
		assert.Equal(t, e, cr.CudaSuccess, "Allocating bytes on device for Projective results failed")

		e = Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, cr.CudaSuccess, "Msm failed")
		outHost := core.HostSliceFromElements[Projective]([]Projective{p})
		outHost.CopyFromDevice(&out)
		out.Free()

		// Check with gnark-crypto
		scalarsFr := make([]fr.Element, size)
		for i, v := range scalars {
			slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
			scalarsFr[i] = slice64
		}

		pointsFp := make([]bn254.G1Affine, size)
		for i, v := range points {
			pointsFp[i] = projectiveToGnarkAffine(v.ToProjective())
		}
		var msmRes bn254.G1Jac
		msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

		var icicleResAsJac bn254.G1Jac
		proj := projectiveToGnarkAffine(outHost[0])
		icicleResAsJac.FromAffine(&proj)

		assert.True(t, msmRes.Equal(&icicleResAsJac))
	}
}

func TestMSMBatch(t *testing.T) {
	cfg := GetDefaultMSMConfig()
	// for _, power := range []int{2, 3, 4, 5, 6, 7, 8, 10, 18} {
	for _, power := range []int{2} {
		for _, batchSize := range []int{3} {
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
				scalarsFr := make([]fr.Element, size)
				for m, v := range scalars[i*size : (i+1)*size] {
					slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
					scalarsFr[m] = slice64
				}

				pointsFp := make([]bn254.G1Affine, size)
				for m, v := range points[i*size : (i+1)*size] {
					pointsFp[m] = projectiveToGnarkAffine(v.ToProjective())
				}

				var msmRes bn254.G1Jac
				msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

				var icicleResAsJac bn254.G1Jac
				proj := projectiveToGnarkAffine(outHost[i])
				icicleResAsJac.FromAffine(&proj)

				assert.True(t, msmRes.Equal(&icicleResAsJac))
			}
		}
	}
}
