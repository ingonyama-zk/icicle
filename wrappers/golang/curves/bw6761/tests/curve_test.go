package tests

import (
	"testing"

	bw6_761 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testAffineZero(suite *suite.Suite) {
	var fieldZero = bw6_761.BaseField{}

	var affineZero bw6_761.Affine
	suite.Equal(affineZero.X, fieldZero)
	suite.Equal(affineZero.Y, fieldZero)

	x := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	y := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var affine bw6_761.Affine
	affine.FromLimbs(x, y)

	affine.Zero()
	suite.Equal(affine.X, fieldZero)
	suite.Equal(affine.Y, fieldZero)
}

func testAffineFromLimbs(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))

	var affine bw6_761.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	suite.ElementsMatch(randLimbs, affine.X.GetLimbs())
	suite.ElementsMatch(randLimbs2, affine.Y.GetLimbs())
}

func testAffineToProjective(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var fieldOne bw6_761.BaseField
	fieldOne.One()

	var expected bw6_761.Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine bw6_761.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	projectivePoint := affine.ToProjective()
	suite.Equal(expected, projectivePoint)
}

func testProjectiveZero(suite *suite.Suite) {
	var projectiveZero bw6_761.Projective
	projectiveZero.Zero()
	var fieldZero = bw6_761.BaseField{}
	var fieldOne bw6_761.BaseField
	fieldOne.One()

	suite.Equal(projectiveZero.X, fieldZero)
	suite.Equal(projectiveZero.Y, fieldOne)
	suite.Equal(projectiveZero.Z, fieldZero)

	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var projective bw6_761.Projective
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

	var projective bw6_761.Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	suite.ElementsMatch(randLimbs, projective.X.GetLimbs())
	suite.ElementsMatch(randLimbs2, projective.Y.GetLimbs())
	suite.ElementsMatch(randLimbs3, projective.Z.GetLimbs())
}

func testProjectiveArithmetic(suite *suite.Suite) {
	points := bw6_761.GenerateProjectivePoints(2)

	point1 := points[0]
	point2 := points[1]

	add := point1.Add(&point2)
	sub := add.Sub(&point2)

	suite.True(point1.ProjectiveEq(&sub))
}

func testProjectiveFromAffine(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	var fieldOne bw6_761.BaseField
	fieldOne.One()

	var expected bw6_761.Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine bw6_761.Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	var projectivePoint bw6_761.Projective
	projectivePoint.FromAffine(affine)
	suite.Equal(expected, projectivePoint)
}

type CurveTestSuite struct {
	suite.Suite
}

func (s *CurveTestSuite) TestCurve() {
	s.Run("TestAffineZero", test_helpers.TestWrapper(&s.Suite, testAffineZero))
	s.Run("TestAffineFromLimbs", test_helpers.TestWrapper(&s.Suite, testAffineFromLimbs))
	s.Run("TestAffineToProjective", test_helpers.TestWrapper(&s.Suite, testAffineToProjective))
	s.Run("TestProjectiveZero", test_helpers.TestWrapper(&s.Suite, testProjectiveZero))
	s.Run("TestProjectiveFromLimbs", test_helpers.TestWrapper(&s.Suite, testProjectiveFromLimbs))
	s.Run("TestProjectiveFromAffine", test_helpers.TestWrapper(&s.Suite, testProjectiveFromAffine))
	s.Run("TestProjectiveArithmetic", test_helpers.TestWrapper(&s.Suite, testProjectiveArithmetic))
}

func TestSuiteCurve(t *testing.T) {
	suite.Run(t, new(CurveTestSuite))
}
