package bn254

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
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

func GenerateG2Points(count int) ([]G2Affine, []bn254.G2Affine) {
	// Declare a slice of integers
	var points []G2Affine
	var pointsAffine []bn254.G2Affine

	// populate the slice
	for i := 0; i < count; i++ {
		gnarkP, _ := randG2Jac()

		var p G2Affine
		p.G2FromG2JacGnark(&gnarkP)

		var gp bn254.G2Affine
		gp.FromJacobian(&gnarkP)
		pointsAffine = append(pointsAffine, gp)
		points = append(points, p)
	}

	return points, pointsAffine
}

func TestMsmG2BN254(t *testing.T) {
	points, gnarkPoints := GenerateG2Points(1)

	assert.Equal(t, points, gnarkPoints)
}
