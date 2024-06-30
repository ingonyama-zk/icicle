package tests

import (
	grumpkin "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/grumpkin"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestAffineZero(t *testing.T) {
	var fieldZero = grumpkin.BaseField{}

	var affineZero grumpkin.Affine
	assert.Equal(t, affineZero.X, fieldZero)
	assert.Equal(t, affineZero.Y, fieldZero)

	x := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	y := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var affine grumpkin.Affine
	affine.FromLimbs(x, y)

	affine.Zero()
	assert.Equal(t, affine.X, fieldZero)
	assert.Equal(t, affine.Y, fieldZero)
}

func TestAffineFromLimbs(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))

	var affine grumpkin.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	assert.ElementsMatch(t, randLimbs, affine.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, affine.Y.GetLimbs())
}

func TestAffineToProjective(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var fieldOne grumpkin.BaseField
	fieldOne.One()

	var expected grumpkin.Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine grumpkin.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	projectivePoint := affine.ToProjective()
	assert.Equal(t, expected, projectivePoint)
}

func TestProjectiveZero(t *testing.T) {
	var projectiveZero grumpkin.Projective
	projectiveZero.Zero()
	var fieldZero = grumpkin.BaseField{}
	var fieldOne grumpkin.BaseField
	fieldOne.One()

	assert.Equal(t, projectiveZero.X, fieldZero)
	assert.Equal(t, projectiveZero.Y, fieldOne)
	assert.Equal(t, projectiveZero.Z, fieldZero)

	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var projective grumpkin.Projective
	projective.FromLimbs(randLimbs, randLimbs, randLimbs)

	projective.Zero()
	assert.Equal(t, projective.X, fieldZero)
	assert.Equal(t, projective.Y, fieldOne)
	assert.Equal(t, projective.Z, fieldZero)
}

func TestProjectiveFromLimbs(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs3 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))

	var projective grumpkin.Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	assert.ElementsMatch(t, randLimbs, projective.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, projective.Y.GetLimbs())
	assert.ElementsMatch(t, randLimbs3, projective.Z.GetLimbs())
}

func TestProjectiveFromAffine(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var fieldOne grumpkin.BaseField
	fieldOne.One()

	var expected grumpkin.Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine grumpkin.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	var projectivePoint grumpkin.Projective
	projectivePoint.FromAffine(affine)
	assert.Equal(t, expected, projectivePoint)
}
