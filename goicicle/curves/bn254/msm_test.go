package bn254

import (
	"fmt"
	"math"
	"math/big"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/ingonyama-zk/icicle/goicicle"
	"github.com/stretchr/testify/assert"
)

func randG1Jac() (bn254.G1Jac, error) {
	var point bn254.G1Jac
	var scalar fr.Element

	_, err := scalar.SetRandom()
	if err != nil {
		return point, err
	}

	genG1Jac, _, _, _ := bn254.Generators()

	//randomBigInt, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 63))
	//randomBigInt, err := rand.Int(rand.Reader, big.NewInt(100))
	randomBigInt := big.NewInt(100)

	point.ScalarMultiplication(&genG1Jac, scalar.BigInt(randomBigInt))
	return point, nil
}

func GeneratePoints(count int) ([]PointAffineNoInfinityBN254, []bn254.G1Affine) {
	// Declare a slice of integers
	var points []PointAffineNoInfinityBN254
	var pointsAffine []bn254.G1Affine

	// populate the slice
	for i := 0; i < 10; i++ {
		gnarkP, _ := randG1Jac()
		var pointAffine bn254.G1Affine
		pointAffine.FromJacobian(&gnarkP)

		p := PointBN254FromJacGnark(&gnarkP).strip_z()

		pointsAffine = append(pointsAffine, pointAffine)
		points = append(points, *p)
	}

	log2_10 := math.Log2(10)
	log2Count := math.Log2(float64(count))
	log2Size := int(math.Ceil(log2Count - log2_10))

	for i := 0; i < log2Size; i++ {
		pointsAffine = append(pointsAffine, pointsAffine...)
		points = append(points, points...)
	}

	return points[:count], pointsAffine[:count]
}

func GeneratePointsProj(count int) ([]PointBN254, []bn254.G1Jac) {
	// Declare a slice of integers
	var points []PointBN254
	var pointsAffine []bn254.G1Jac

	// Use a loop to populate the slice
	for i := 0; i < count; i++ {
		gnarkP, _ := randG1Jac()
		p := PointBN254FromJacGnark(&gnarkP)

		pointsAffine = append(pointsAffine, gnarkP)
		points = append(points, *p)
	}

	return points, pointsAffine
}

func GenerateScalars(count int) ([]ScalarField, []fr.Element) {
	// Declare a slice of integers
	var scalars []ScalarField
	var scalars_fr []fr.Element

	var rand fr.Element
	for i := 0; i < 10; i++ {
		rand.SetRandom()
		s := NewFieldFromFrGnark[ScalarField](rand)

		scalars_fr = append(scalars_fr, rand)
		scalars = append(scalars, *s)
	}

	log2_10 := math.Log2(10)
	log2Count := math.Log2(float64(count))
	log2Size := int(math.Ceil(log2Count - log2_10))

	for i := 0; i < log2Size; i++ {
		scalars_fr = append(scalars_fr, scalars_fr...)
		scalars = append(scalars, scalars...)
	}

	return scalars[:count], scalars_fr[:count]
}

func TestMSM(t *testing.T) {
	for _, v := range []int{24} {
		count := 1 << v

		points, gnarkPoints := GeneratePoints(count)
		fmt.Print("Finished generating points\n")
		scalars, gnarkScalars := GenerateScalars(count)
		fmt.Print("Finished generating scalars\n")

		out := new(PointBN254)
		startTime := time.Now()
		_, e := MsmBN254(out, points, scalars, 0) // non mont
		fmt.Printf("icicle MSM took: %d ms\n", time.Since(startTime).Milliseconds())

		assert.Equal(t, e, nil, "error should be nil")
		fmt.Print("Finished icicle MSM\n")

		var bn254AffineLib bn254.G1Affine

		gResult, _ := bn254AffineLib.MultiExp(gnarkPoints, gnarkScalars, ecc.MultiExpConfig{})
		fmt.Print("Finished Gnark MSM\n")

		assert.Equal(t, out.toGnarkAffine(), gResult)
	}
}

func TestCommitMSM(t *testing.T) {
	for _, _ = range []int{24} {
		count := 12_180_757
		// count := 1 << v - 1

		points, gnarkPoints := GeneratePoints(count)
		fmt.Print("Finished generating points\n")
		scalars, gnarkScalars := GenerateScalars(count)
		fmt.Print("Finished generating scalars\n")

		out_d, _ := goicicle.CudaMalloc(96)

		pointsBytes := count * 64
		points_d, _ := goicicle.CudaMalloc(pointsBytes)
		goicicle.CudaMemCpyHtoD[PointAffineNoInfinityBN254](points_d, points, pointsBytes)

		scalarBytes := count * 32
		scalars_d, _ := goicicle.CudaMalloc(scalarBytes)
		goicicle.CudaMemCpyHtoD[ScalarField](scalars_d, scalars, scalarBytes)

		startTime := time.Now()
		e := Commit(out_d, scalars_d, points_d, count)
		fmt.Printf("icicle MSM took: %d ms\n", time.Since(startTime).Milliseconds())

		outHost := make([]PointBN254, 1)
		goicicle.CudaMemCpyDtoH[PointBN254](outHost, out_d, 96)

		assert.Equal(t, e, 0, "error should be 0")
		fmt.Print("Finished icicle MSM\n")

		var bn254AffineLib bn254.G1Affine

		gResult, _ := bn254AffineLib.MultiExp(gnarkPoints, gnarkScalars, ecc.MultiExpConfig{})
		fmt.Print("Finished Gnark MSM\n")

		assert.Equal(t, outHost[0].toGnarkAffine(), gResult)
	}
}

func TestBenchMSM(t *testing.T) {
	for _, batchPow2 := range []int{2, 4} {
		for _, pow2 := range []int{4, 6} {
			msmSize := 1 << pow2
			batchSize := 1 << batchPow2
			count := msmSize * batchSize

			points, _ := GeneratePoints(count)
			scalars, _ := GenerateScalars(count)

			a, e := MsmBatchBN254(&points, &scalars, batchSize, 0)

			if e != nil {
				t.Errorf("MsmBatchBN254 returned an error: %v", e)
			}

			if len(a) != batchSize {
				t.Errorf("Expected length %d, but got %d", batchSize, len(a))
			}
		}
	}
}

func BenchmarkMSM(b *testing.B) {
	LOG_MSM_SIZES := []int{20, 21, 22, 23, 24, 25, 26}

	for _, logMsmSize := range LOG_MSM_SIZES {
		msmSize := 1 << logMsmSize
		points, _ := GeneratePoints(msmSize)
		scalars, _ := GenerateScalars(msmSize)
		b.Run(fmt.Sprintf("MSM %d", logMsmSize), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				out := new(PointBN254)
				_, e := MsmBN254(out, points, scalars, 0)

				if e != nil {
					panic("Error occured")
				}
			}
		})
	}
}
