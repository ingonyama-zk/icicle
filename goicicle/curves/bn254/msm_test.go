package bn254

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func randG1Jac() (bn254.G1Jac, error) {
	var point bn254.G1Jac
	var scalar fr.Element

	_, err := scalar.SetRandom()
	if err != nil {
		return point, err
	}

	genG1Jac, _, _, _ := bn254.Generators()

	randomBigInt, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		panic(err)
	}

	point.ScalarMultiplication(&genG1Jac, scalar.BigInt(randomBigInt))
	return point, nil
}

func GeneratePoints(count int) ([]PointAffineNoInfinityBN254, []bn254.G1Affine) {
	// Declare a slice of integers
	var points []PointAffineNoInfinityBN254
	var pointsAffine []bn254.G1Affine

	// Use a loop to populate the slice
	for i := 0; i < count; i++ {
		gnarkP, _ := randG1Jac()
		var pointAffine bn254.G1Affine
		pointAffine.FromJacobian(&gnarkP)

		p := PointBN254FromGnark(&gnarkP).strip_z()

		pointsAffine = append(pointsAffine, pointAffine)
		points = append(points, *p)
	}

	return points, pointsAffine
}

func GenerateScalars(count int) ([]FieldBN254, []fr.Element) {
	// Declare a slice of integers
	var scalars []FieldBN254
	var scalars_fr []fr.Element

	var rand fr.Element

	// Generate a random fr.Element

	// Use a loop to populate the slice
	for i := 0; i < count; i++ {
		rand.SetRandom()
		s := FieldBN254FromGnark(rand)

		scalars_fr = append(scalars_fr, rand)
		scalars = append(scalars, *s)
	}

	return scalars, scalars_fr
}

func TestMSM(t *testing.T) {
	for _, v := range []int{6, 9} {
		count := 1 << v

		points, gnarkPoints := GeneratePoints(count)
		scalars, gnarkScalars := GenerateScalars(count)

		a, e := MsmBN254(points, scalars, 0)

		assert.Equal(t, e, nil, "error should be nil")

		var bn254AffineLib bn254.G1Affine

		a1, _ := bn254AffineLib.MultiExp(gnarkPoints, gnarkScalars, ecc.MultiExpConfig{})
		assert.Equal(t, a.toGnarkAffine(), a1)
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
		b.Run(fmt.Sprintf("MSM %d", logMsmSize), func(b *testing.B) {
			points, _ := GeneratePoints(msmSize)
			scalars, _ := GenerateScalars(msmSize)
			for n := 0; n < b.N; n++ {
				_, e := MsmBN254(points, scalars, 0)

				if e != nil {
					panic("Error occured")
				}
			}
		})
	}
}
