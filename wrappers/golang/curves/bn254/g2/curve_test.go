package g2

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestG2AffineZero(t *testing.T) {
	var fieldZero = G2BaseField{}

	var affineZero G2Affine
	assert.Equal(t, affineZero.X, fieldZero)
	assert.Equal(t, affineZero.Y, fieldZero)

	x := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	y := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var affine G2Affine
	affine.FromLimbs(x, y)

	affine.Zero()
	assert.Equal(t, affine.X, fieldZero)
	assert.Equal(t, affine.Y, fieldZero)
}

func TestG2AffineFromLimbs(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))

	var affine G2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	assert.ElementsMatch(t, randLimbs, affine.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, affine.Y.GetLimbs())
}

func TestG2AffineToProjective(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var fieldOne G2BaseField
	fieldOne.One()

	var expected G2Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.limbs[:])

	var affine G2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	projectivePoint := affine.ToProjective()
	assert.Equal(t, expected, projectivePoint)
}

func TestG2ProjectiveZero(t *testing.T) {
	var projectiveZero G2Projective
	projectiveZero.Zero()
	var fieldZero = G2BaseField{}
	var fieldOne G2BaseField
	fieldOne.One()

	assert.Equal(t, projectiveZero.X, fieldZero)
	assert.Equal(t, projectiveZero.Y, fieldOne)
	assert.Equal(t, projectiveZero.Z, fieldZero)

	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var projective G2Projective
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

	var projective G2Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	assert.ElementsMatch(t, randLimbs, projective.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, projective.Y.GetLimbs())
	assert.ElementsMatch(t, randLimbs3, projective.Z.GetLimbs())
}

func TestG2ProjectiveFromAffine(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var fieldOne G2BaseField
	fieldOne.One()

	var expected G2Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.limbs[:])

	var affine G2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	var projectivePoint G2Projective
	projectivePoint.FromAffine(affine)
	assert.Equal(t, expected, projectivePoint)
}
