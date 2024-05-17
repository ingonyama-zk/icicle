package tests

import (
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/g2"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestG2AffineZero(t *testing.T) {
	var fieldZero = g2.G2BaseField{}

	var affineZero g2.G2Affine
	assert.Equal(t, affineZero.X, fieldZero)
	assert.Equal(t, affineZero.Y, fieldZero)

	x := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	y := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var affine g2.G2Affine
	affine.FromLimbs(x, y)

	affine.Zero()
	assert.Equal(t, affine.X, fieldZero)
	assert.Equal(t, affine.Y, fieldZero)
}

func TestG2AffineFromLimbs(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))

	var affine g2.G2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	assert.ElementsMatch(t, randLimbs, affine.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, affine.Y.GetLimbs())
}

func TestG2AffineToProjective(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var fieldOne g2.G2BaseField
	fieldOne.One()

	var expected g2.G2Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine g2.G2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	projectivePoint := affine.ToProjective()
	assert.Equal(t, expected, projectivePoint)
}

func TestG2ProjectiveZero(t *testing.T) {
	var projectiveZero g2.G2Projective
	projectiveZero.Zero()
	var fieldZero = g2.G2BaseField{}
	var fieldOne g2.G2BaseField
	fieldOne.One()

	assert.Equal(t, projectiveZero.X, fieldZero)
	assert.Equal(t, projectiveZero.Y, fieldOne)
	assert.Equal(t, projectiveZero.Z, fieldZero)

	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var projective g2.G2Projective
	projective.FromLimbs(randLimbs, randLimbs, randLimbs)

	projective.Zero()
	assert.Equal(t, projective.X, fieldZero)
	assert.Equal(t, projective.Y, fieldOne)
	assert.Equal(t, projective.Z, fieldZero)
}

func TestG2ProjectiveFromLimbs(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs3 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))

	var projective g2.G2Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	assert.ElementsMatch(t, randLimbs, projective.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, projective.Y.GetLimbs())
	assert.ElementsMatch(t, randLimbs3, projective.Z.GetLimbs())
}

func TestG2ProjectiveFromAffine(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var fieldOne g2.G2BaseField
	fieldOne.One()

	var expected g2.G2Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine g2.G2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	var projectivePoint g2.G2Projective
	projectivePoint.FromAffine(affine)
	assert.Equal(t, expected, projectivePoint)
}
