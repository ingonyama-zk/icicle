package internal

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestAffineZero(t *testing.T) {
	var fieldZero = MockField{}

	var affineZero Affine
	affineZero.Zero()

	assert.Equal(t, affineZero.X, fieldZero)
	assert.Equal(t, affineZero.Y, fieldZero)

	var fieldZeroOfLength = MockField{}

	affine := Affine{
		X: MockField{
			limbs: [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		Y: MockField{
			limbs: [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8},
		},
	}

	affine.Zero()
	assert.Equal(t, affine.X, fieldZeroOfLength)
	assert.Equal(t, affine.Y, fieldZeroOfLength)
}

func TestAffineFromLimbs(t *testing.T) {
	randLimbs := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	randLimbs2 := []uint32{11, 12, 13, 14, 15, 16, 17, 18}

	var affine Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	assert.ElementsMatch(t, randLimbs, affine.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, affine.Y.GetLimbs())
}

func TestAffineToProjective(t *testing.T) {
	randLimbs := [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8}
	randLimbs2 := [BASE_LIMBS]uint32{11, 12, 13, 14, 15, 16, 17, 18}
	var fieldOne MockField
	limbsOne := []uint32{1, 0, 0, 0, 0, 0, 0, 0}
	fieldOne.FromLimbs(limbsOne)

	expected := Projective{
		X: MockField{
			limbs: randLimbs,
		},
		Y: MockField{
			limbs: randLimbs2,
		},
		Z: fieldOne,
	}

	var affine Affine
	affine.FromLimbs(randLimbs[:], randLimbs2[:])

	projectivePoint := affine.ToProjective()
	assert.Equal(t, expected, projectivePoint)
}

func TestProjectiveZero(t *testing.T) {
	var fieldZero = MockField{}

	var projectiveZero Projective
	projectiveZero.Zero()

	assert.Equal(t, projectiveZero.X, fieldZero)
	assert.Equal(t, projectiveZero.Y, fieldZero)

	var fieldZeroOfLength = MockField{
		limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0},
	}

	projective := Projective{
		X: MockField{
			limbs: [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		Y: MockField{
			limbs: [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		Z: MockField{
			limbs: [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8},
		},
	}

	projective.Zero()
	assert.Equal(t, projective.X, fieldZeroOfLength)
	assert.Equal(t, projective.Y, fieldZeroOfLength)
	assert.Equal(t, projective.Z, fieldZeroOfLength)
}

func TestProjectiveFromLimbs(t *testing.T) {
	randLimbs := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	randLimbs2 := []uint32{11, 12, 13, 14, 15, 16, 17, 18}
	randLimbs3 := []uint32{21, 22, 23, 24, 25, 26, 27, 28}

	var projective Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	assert.ElementsMatch(t, randLimbs, projective.X.GetLimbs())
	assert.ElementsMatch(t, randLimbs2, projective.Y.GetLimbs())
	assert.ElementsMatch(t, randLimbs3, projective.Z.GetLimbs())
}

func TestProjectiveFromAffine(t *testing.T) {
	randLimbs := [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8}
	randLimbs2 := [BASE_LIMBS]uint32{11, 12, 13, 14, 15, 16, 17, 18}
	var fieldOne MockField
	limbsOne := []uint32{1, 0, 0, 0, 0, 0, 0, 0}
	fieldOne.FromLimbs(limbsOne)

	expected := Projective{
		X: MockField{
			limbs: randLimbs,
		},
		Y: MockField{
			limbs: randLimbs2,
		},
		Z: fieldOne,
	}

	var affine Affine
	affine.FromLimbs(randLimbs[:], randLimbs2[:])

	var projectivePoint Projective
	projectivePoint.FromAffine(affine)
	assert.Equal(t, expected, projectivePoint)
}
