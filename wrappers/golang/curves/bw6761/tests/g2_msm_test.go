package tests

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bw6-761"
	"github.com/consensys/gnark-crypto/ecc/bw6-761/fp"
	"github.com/consensys/gnark-crypto/ecc/bw6-761/fr"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	icicleBw6_761 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761/g2"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func projectiveToGnarkAffineG2(p g2.G2Projective) bw6761.G2Affine {
	px, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)((&p.X).ToBytesLittleEndian()))
	py, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)((&p.Y).ToBytesLittleEndian()))
	pz, _ := fp.LittleEndian.Element((*[fp.Bytes]byte)((&p.Z).ToBytesLittleEndian()))

	zInv := new(fp.Element)
	x := new(fp.Element)
	y := new(fp.Element)

	zInv.Inverse(&pz)

	x.Mul(&px, zInv)
	y.Mul(&py, zInv)

	return bw6761.G2Affine{X: *x, Y: *y}
}

func testAgainstGnarkCryptoMsmG2(t *testing.T, scalars core.HostSlice[icicleBw6_761.ScalarField], points core.HostSlice[g2.G2Affine], out g2.G2Projective) {
	scalarsFr := make([]fr.Element, len(scalars))
	for i, v := range scalars {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		scalarsFr[i] = slice64
	}

	pointsFp := make([]bw6761.G2Affine, len(points))
	for i, v := range points {
		pointsFp[i] = projectiveToGnarkAffineG2(v.ToProjective())
	}

	testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(t, scalarsFr, pointsFp, out)
}

func testAgainstGnarkCryptoMsmG2GnarkCryptoTypes(t *testing.T, scalarsFr core.HostSlice[fr.Element], pointsFp core.HostSlice[bw6761.G2Affine], out g2.G2Projective) {
	var msmRes bw6761.G2Jac
	msmRes.MultiExp(pointsFp, scalarsFr, ecc.MultiExpConfig{})

	var msmResAffine bw6761.G2Affine
	msmResAffine.FromJacobian(&msmRes)

	icicleResAffine := projectiveToGnarkAffineG2(out)

	assert.Equal(t, msmResAffine, icicleResAffine)
}

func convertIcicleAffineToG2Affine(iciclePoints []g2.G2Affine) []bw6761.G2Affine {
	points := make([]bw6761.G2Affine, len(iciclePoints))
	for index, iciclePoint := range iciclePoints {
		xBytes := ([fp.Bytes]byte)(iciclePoint.X.ToBytesLittleEndian())
		fpXElem, _ := fp.LittleEndian.Element(&xBytes)

		yBytes := ([fp.Bytes]byte)(iciclePoint.Y.ToBytesLittleEndian())
		fpYElem, _ := fp.LittleEndian.Element(&yBytes)
		points[index] = bw6761.G2Affine{
			X: fpXElem,
			Y: fpYElem,
		}
	}

	return points
}

func TestMSMG2(t *testing.T) {
	cfg := g2.G2GetDefaultMSMConfig()
	cfg.IsAsync = true
	for _, power := range []int{2, 3, 4, 5, 6} {
		runtime.SetDevice(&DEVICE)
		size := 1 << power

		scalars := icicleBw6_761.GenerateScalars(size)
		points := g2.G2GenerateAffinePoints(size)

		stream, _ := runtime.CreateStream()
		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.MallocAsync(p.Size(), 1, stream)
		assert.Equal(t, e, runtime.Success, "Allocating bytes on device for Projective results failed")
		cfg.StreamHandle = stream

		e = g2.G2Msm(scalars, points, &cfg, out)
		assert.Equal(t, e, runtime.Success, "Msm failed")
		outHost := make(core.HostSlice[g2.G2Projective], 1)
		outHost.CopyFromDeviceAsync(&out, stream)
		out.FreeAsync(stream)

		runtime.SynchronizeStream(stream)
		runtime.DestroyStream(stream)
		// Check with gnark-crypto
		testAgainstGnarkCryptoMsmG2(t, scalars, points, outHost[0])
	}
}

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
		pointsGnark := convertIcicleAffineToG2Affine(points)
		pointsHost := (core.HostSlice[bw6761.G2Affine])(pointsGnark)

		var p g2.G2Projective
		var out core.DeviceSlice
		_, e := out.Malloc(p.Size(), 1)
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
	for _, power := range []int{5, 6} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)
			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBw6_761.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(totalSize)

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e := out.Malloc(p.Size(), batchSize)
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
	if DEVICE.GetDeviceType() == "CPU" {
		t.Skip("Skipping cpu test")
	}
	cfg := g2.G2GetDefaultMSMConfig()
	const precomputeFactor = 8
	cfg.PrecomputeFactor = precomputeFactor

	for _, power := range []int{7, 8} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)

			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBw6_761.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(totalSize)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			assert.Equal(t, runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			cfg.BatchSize = int32(batchSize)
			cfg.ArePointsSharedInBatch = false
			e = g2.G2PrecomputeBases(points, &cfg, precomputeOut)
			assert.Equal(t, runtime.Success, e, "PrecomputeBases failed")

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(p.Size(), batchSize)
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

	for _, power := range []int{4, 5, 6} {
		for _, batchSize := range []int{1, 3, 5} {
			runtime.SetDevice(&DEVICE)

			size := 1 << power
			totalSize := size * batchSize
			scalars := icicleBw6_761.GenerateScalars(totalSize)
			points := g2.G2GenerateAffinePoints(size)

			var precomputeOut core.DeviceSlice
			_, e := precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))
			assert.Equal(t, runtime.Success, e, "Allocating bytes on device for PrecomputeBases results failed")

			e = g2.G2PrecomputeBases(points, &cfg, precomputeOut)
			assert.Equal(t, runtime.Success, e, "PrecomputeBases failed")

			var p g2.G2Projective
			var out core.DeviceSlice
			_, e = out.Malloc(p.Size(), batchSize)
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
	for _, power := range []int{2, 3, 4, 5} {
		runtime.SetDevice(&DEVICE)

		size := 1 << power

		scalars := icicleBw6_761.GenerateScalars(size)
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
			for _, power := range []int{2, 3, 4, 5, 6} {
				size := 1 << power
				scalars := icicleBw6_761.GenerateScalars(size)
				points := g2.G2GenerateAffinePoints(size)

				stream, _ := runtime.CreateStream()
				var p g2.G2Projective
				var out core.DeviceSlice
				_, e := out.MallocAsync(p.Size(), 1, stream)
				assert.Equal(t, e, runtime.Success, "Allocating bytes on device for Projective results failed")
				cfg.StreamHandle = stream

				e = g2.G2Msm(scalars, points, &cfg, out)
				assert.Equal(t, e, runtime.Success, "Msm failed")
				outHost := make(core.HostSlice[g2.G2Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				runtime.SynchronizeStream(stream)
				runtime.DestroyStream(stream)
				// Check with gnark-crypto
				testAgainstGnarkCryptoMsmG2(t, scalars, points, outHost[0])
			}
		})
	}
	wg.Wait()
}
