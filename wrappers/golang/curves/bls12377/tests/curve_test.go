package tests

import (
	"testing"

	bls12_377 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testAffineZero(suite *suite.Suite) {
	var fieldZero = bls12_377.BaseField{}

	var affineZero bls12_377.Affine
	suite.Equal(affineZero.X, fieldZero)
	suite.Equal(affineZero.Y, fieldZero)

	x := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	y := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var affine bls12_377.Affine
	affine.FromLimbs(x, y)

	affine.Zero()
	suite.Equal(affine.X, fieldZero)
	suite.Equal(affine.Y, fieldZero)
}

func testAffineFromLimbs(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))

	var affine bls12_377.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	suite.ElementsMatch(randLimbs, affine.X.GetLimbs())
	suite.ElementsMatch(randLimbs2, affine.Y.GetLimbs())
}

func testAffineToProjective(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var fieldOne bls12_377.BaseField
	fieldOne.One()

	var expected bls12_377.Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine bls12_377.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	projectivePoint := affine.ToProjective()
	suite.Equal(expected, projectivePoint)
}

func testProjectiveZero(suite *suite.Suite) {
	var projectiveZero bls12_377.Projective
	projectiveZero.Zero()
	var fieldZero = bls12_377.BaseField{}
	var fieldOne bls12_377.BaseField
	fieldOne.One()

	suite.Equal(projectiveZero.X, fieldZero)
	suite.Equal(projectiveZero.Y, fieldOne)
	suite.Equal(projectiveZero.Z, fieldZero)

	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var projective bls12_377.Projective
	projective.FromLimbs(randLimbs, randLimbs, randLimbs)

	projective.Zero()
	suite.Equal(projective.X, fieldZero)
	suite.Equal(projective.Y, fieldOne)
	suite.Equal(projective.Z, fieldZero)
}

func testProjectiveFromLimbs(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs3 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))

	var projective bls12_377.Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	suite.ElementsMatch(randLimbs, projective.X.GetLimbs())
	suite.ElementsMatch(randLimbs2, projective.Y.GetLimbs())
	suite.ElementsMatch(randLimbs3, projective.Z.GetLimbs())
}

func testProjectiveArithmetic(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs3 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))

	var point bls12_377.Projective
	point.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	randLimbs = test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 = test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs3 = test_helpers.GenerateRandomLimb(int(BASE_LIMBS))

	var point2 bls12_377.Projective
	point2.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	add := point.Add(&point2)
	sub := add.Sub(&point2)

	suite.Equal(sub.X, point.X)
}

func testProjectiveFromAffine(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var fieldOne bls12_377.BaseField
	fieldOne.One()

	var expected bls12_377.Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine bls12_377.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	var projectivePoint bls12_377.Projective
	projectivePoint.FromAffine(affine)
	suite.Equal(expected, projectivePoint)
}

type CurveTestSuite struct {
	suite.Suite
}

func (s *CurveTestSuite) TestCurve() {
	s.Run("TestAffineZero", testWrapper(&s.Suite, testAffineZero))
	s.Run("TestAffineFromLimbs", testWrapper(&s.Suite, testAffineFromLimbs))
	s.Run("TestAffineToProjective", testWrapper(&s.Suite, testAffineToProjective))
	s.Run("TestProjectiveZero", testWrapper(&s.Suite, testProjectiveZero))
	s.Run("TestProjectiveFromLimbs", testWrapper(&s.Suite, testProjectiveFromLimbs))
	s.Run("TestProjectiveFromAffine", testWrapper(&s.Suite, testProjectiveFromAffine))
	s.Run("TestProjectiveArithmetic", testWrapper(&s.Suite, testProjectiveArithmetic))
}

func TestSuiteCurve(t *testing.T) {
	suite.Run(t, new(CurveTestSuite))
}
