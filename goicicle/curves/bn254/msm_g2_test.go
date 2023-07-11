package bn254

import (
	"fmt"
	"math/big"
	"testing"
	"time"
	"unsafe"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/ingonyama-zk/icicle/goicicle"
	"github.com/stretchr/testify/assert"
)

func randG2Jac() (bn254.G2Jac, error) {
	var point bn254.G2Jac
	var scalar fr.Element

	_, err := scalar.SetRandom()
	if err != nil {
		return point, err
	}

	_, genG2Jac, _, _ := bn254.Generators()

	randomBigInt := big.NewInt(1000)

	point.ScalarMultiplication(&genG2Jac, scalar.BigInt(randomBigInt))
	return point, nil
}

func GenerateG2Points(count int) ([]G2PointAffine, []bn254.G2Affine) {
	// Declare a slice of integers
	var points []G2PointAffine
	var pointsAffine []bn254.G2Affine

	// populate the slice
	for i := 0; i < count; i++ {
		gnarkP, _ := randG2Jac()

		var p G2PointAffine
		p.FromGnarkJac(&gnarkP)

		var gp bn254.G2Affine
		gp.FromJacobian(&gnarkP)
		pointsAffine = append(pointsAffine, gp)
		points = append(points, p)
	}

	return points, pointsAffine
}

func TestMsmG2BN254(t *testing.T) {
	for _, v := range []int{24} {
		count := 1 << v
		points, gnarkPoints := GenerateG2Points(count)
		fmt.Print("Finished generating points\n")
		scalars, gnarkScalars := GenerateScalars(count)
		fmt.Print("Finished generating scalars\n")

		out := new(G2Point)
		_, e := MsmG2BN254(out, points, scalars, 0)
		assert.Equal(t, e, nil, "error should be nil")

		var result G2PointAffine
		var bn254AffineLib bn254.G2Affine

		gResult, _ := bn254AffineLib.MultiExp(gnarkPoints, gnarkScalars, ecc.MultiExpConfig{})

		result.FromGnarkAffine(gResult)

		pp := result.ToProjective()
		assert.True(t, out.eqg2(&pp))
		//assert.Equal(t, out, result.ToProjective())
	}
}

func BenchmarkMsmG2BN254(b *testing.B) {
	LOG_MSM_SIZES := []int{20, 21, 22, 23, 24, 25, 26}

	for _, logMsmSize := range LOG_MSM_SIZES {
		msmSize := 1 << logMsmSize
		points, _ := GenerateG2Points(msmSize)
		scalars, _ := GenerateScalars(msmSize)
		b.Run(fmt.Sprintf("MSM G2 %d", logMsmSize), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				out := new(G2Point)
				_, e := MsmG2BN254(out, points, scalars, 0)

				if e != nil {
					panic("Error occured")
				}
			}
		})
	}
}

func TestCommitG2MSM(t *testing.T) {
	for _, v := range []int{24} {
		count := 1 << v

		points, gnarkPoints := GenerateG2Points(count)
		fmt.Print("Finished generating points\n")
		scalars, gnarkScalars := GenerateScalars(count)
		fmt.Print("Finished generating scalars\n")

		out_d, _ := goicicle.CudaMalloc(128)

		var sizeCheckG2PointAffine G2PointAffine
		pointsBytes := count * int(unsafe.Sizeof(sizeCheckG2PointAffine))

		points_d, _ := goicicle.CudaMalloc(pointsBytes)
		goicicle.CudaMemCpyHtoD[G2PointAffine](points_d, points, pointsBytes)

		scalarBytes := count * 32
		scalars_d, _ := goicicle.CudaMalloc(scalarBytes)
		goicicle.CudaMemCpyHtoD[ScalarField](scalars_d, scalars, scalarBytes)

		startTime := time.Now()
		e := CommitG2(out_d, scalars_d, points_d, count)
		fmt.Printf("icicle MSM took: %d ms\n", time.Since(startTime).Milliseconds())

		outHost := make([]G2PointAffine, 1)
		goicicle.CudaMemCpyDtoH[G2PointAffine](outHost, out_d, 128)

		assert.Equal(t, e, 0, "error should be 0")
		fmt.Print("Finished icicle MSM\n")

		var bn254AffineLib bn254.G2Affine

		gResult, _ := bn254AffineLib.MultiExp(gnarkPoints, gnarkScalars, ecc.MultiExpConfig{})
		fmt.Print("Finished Gnark MSM\n")
		var result G2PointAffine
		result.FromGnarkAffine(gResult)

		assert.Equal(t, outHost[0], result)
	}
}

func TestBatchG2MSM(t *testing.T) {
	for _, batchPow2 := range []int{2, 4} {
		for _, pow2 := range []int{4, 6} {
			msmSize := 1 << pow2
			batchSize := 1 << batchPow2
			count := msmSize * batchSize

			points, _ := GenerateG2Points(count)
			scalars, _ := GenerateScalars(count)

			a, e := MsmG2BatchBN254(&points, &scalars, batchSize, 0)

			if e != nil {
				t.Errorf("MsmBatchBN254 returned an error: %v", e)
			}

			if len(a) != batchSize {
				t.Errorf("Expected length %d, but got %d", batchSize, len(a))
			}
		}
	}
}
