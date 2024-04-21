package g2

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_g2AffineZero(t *testing.T) {
	var fieldZero = _g2BaseField{}

	var affineZero _g2Affine
	assert.Equal(t, affineZero.X, fieldZero)
	assert.Equal(t, affineZero.Y, fieldZero)

	x := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	y := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	var affine _g2Affine
	affine.FromLimbs(x, y)

	affine.Zero()
	assert.Equal(t, affine.X, fieldZero)
	assert.Equal(t, affine.Y, fieldZero)
}

func Test_g2AffineFromLimbs(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))

	var affine _g2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	assert.ElementsMatch(t, randLimbs, affine.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, affine.Y.GetLimbs())
}

func Test_g2AffineToProjective(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	var fieldOne _g2BaseField
	fieldOne.One()

	var expected _g2Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.limbs[:])

	var affine _g2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	projectivePoint := affine.ToProjective()
	assert.Equal(t, expected, projectivePoint)
}

func Test_g2ProjectiveZero(t *testing.T) {
	var projectiveZero _g2Projective
	projectiveZero.Zero()
	var fieldZero = _g2BaseField{}
	var fieldOne _g2BaseField
	fieldOne.One()

	assert.Equal(t, projectiveZero.X, fieldZero)
	assert.Equal(t, projectiveZero.Y, fieldOne)
	assert.Equal(t, projectiveZero.Z, fieldZero)

	randLimbs := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	var projective _g2Projective
	projective.FromLimbs(randLimbs, randLimbs, randLimbs)

	projective.Zero()
	assert.Equal(t, projective.X, fieldZero)
	assert.Equal(t, projective.Y, fieldOne)
	assert.Equal(t, projective.Z, fieldZero)
}

func Test_g2ProjectiveFromLimbs(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	randLimbs3 := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))

	var projective _g2Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	assert.ElementsMatch(t, randLimbs, projective.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, projective.Y.GetLimbs())
	assert.ElementsMatch(t, randLimbs3, projective.Z.GetLimbs())
}

func Test_g2ProjectiveFromAffine(t *testing.T) {
	randLimbs := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(_g2BASE_LIMBS))
	var fieldOne _g2BaseField
	fieldOne.One()

	var expected _g2Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.limbs[:])

	var affine _g2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	var projectivePoint _g2Projective
	projectivePoint.FromAffine(affine)
	assert.Equal(t, expected, projectivePoint)
}
