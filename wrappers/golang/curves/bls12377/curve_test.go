package bls12377

import (
	"testing"
	"github.com/stretchr/testify/assert"
)

func TestAffineZero(t *testing.T) {
	var fieldZero = BaseField{}

	var affineZero Affine
	assert.Equal(t, affineZero.X, fieldZero)
	assert.Equal(t, affineZero.Y, fieldZero)

	x := generateRandomLimb(int(BASE_LIMBS))
	y := generateRandomLimb(int(BASE_LIMBS))
	var affine Affine
	affine.FromLimbs(x, y)

	affine.Zero()
	assert.Equal(t, affine.X, fieldZero)
	assert.Equal(t, affine.Y, fieldZero)
}

func TestAffineFromLimbs(t *testing.T) {
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := generateRandomLimb(int(BASE_LIMBS))

	var affine Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	assert.ElementsMatch(t, randLimbs, affine.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, affine.Y.GetLimbs())
}

func TestAffineToProjective(t *testing.T) {
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := generateRandomLimb(int(BASE_LIMBS))
	var fieldOne BaseField
	fieldOne.One()

	var expected Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.limbs[:])

	var affine Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	projectivePoint := affine.ToProjective()
	assert.Equal(t, expected, projectivePoint)
}

func TestProjectiveZero(t *testing.T) {
	var fieldZero = BaseField{}

	var projectiveZero Projective
	projectiveZero.Zero()

	assert.Equal(t, projectiveZero.X, fieldZero)
	assert.Equal(t, projectiveZero.Y, fieldZero)

	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	var projective Projective
	projective.FromLimbs(randLimbs, randLimbs, randLimbs)

	projective.Zero()
	assert.Equal(t, projective.X, fieldZero)
	assert.Equal(t, projective.Y, fieldZero)
	assert.Equal(t, projective.Z, fieldZero)
}

func TestProjectiveFromLimbs(t *testing.T) {
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := generateRandomLimb(int(BASE_LIMBS))
	randLimbs3 := generateRandomLimb(int(BASE_LIMBS))

	var projective Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	assert.ElementsMatch(t, randLimbs, projective.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, projective.Y.GetLimbs())
	assert.ElementsMatch(t, randLimbs3, projective.Z.GetLimbs())
}

func TestProjectiveFromAffine(t *testing.T) {
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := generateRandomLimb(int(BASE_LIMBS))
	var fieldOne BaseField
	fieldOne.One()

	var expected Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.limbs[:])

	var affine Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	var projectivePoint Projective
	projectivePoint.FromAffine(affine)
	assert.Equal(t, expected, projectivePoint)
}
