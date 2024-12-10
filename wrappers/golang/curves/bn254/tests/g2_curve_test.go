package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/g2"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testG2AffineZero(suite *suite.Suite) {
	var fieldZero = g2.G2BaseField{}

	var affineZero g2.G2Affine
	suite.Equal(affineZero.X, fieldZero)
	suite.Equal(affineZero.Y, fieldZero)

	x := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	y := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var affine g2.G2Affine
	affine.FromLimbs(x, y)

	affine.Zero()
	suite.Equal(affine.X, fieldZero)
	suite.Equal(affine.Y, fieldZero)
}

func testG2AffineFromLimbs(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))

	var affine g2.G2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	suite.ElementsMatch(randLimbs, affine.X.GetLimbs())
	suite.ElementsMatch(randLimbs2, affine.Y.GetLimbs())
}

func testG2AffineToProjective(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var fieldOne g2.G2BaseField
	fieldOne.One()

	var expected g2.G2Projective
	expected.FromLimbs(randLimbs, randLimbs2, fieldOne.GetLimbs()[:])

	var affine g2.G2Affine
	affine.FromLimbs(randLimbs, randLimbs2)

	projectivePoint := affine.ToProjective()
	suite.Equal(expected, projectivePoint)
}

func testG2ProjectiveZero(suite *suite.Suite) {
	var projectiveZero g2.G2Projective
	projectiveZero.Zero()
	var fieldZero = g2.G2BaseField{}
	var fieldOne g2.G2BaseField
	fieldOne.One()

	suite.Equal(projectiveZero.X, fieldZero)
	suite.Equal(projectiveZero.Y, fieldOne)
	suite.Equal(projectiveZero.Z, fieldZero)

	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	var projective g2.G2Projective
	projective.FromLimbs(randLimbs, randLimbs, randLimbs)

	projective.Zero()
	suite.Equal(projective.X, fieldZero)
	suite.Equal(projective.Y, fieldOne)
	suite.Equal(projective.Z, fieldZero)
}

func testG2ProjectiveFromLimbs(suite *suite.Suite) {
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs2 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	randLimbs3 := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))

	var projective g2.G2Projective
	projective.FromLimbs(randLimbs, randLimbs2, randLimbs3)

	suite.ElementsMatch(randLimbs, projective.X.GetLimbs())
	suite.ElementsMatch(randLimbs2, projective.Y.GetLimbs())
	suite.ElementsMatch(randLimbs3, projective.Z.GetLimbs())
}

func testG2ProjectiveArithmetic(suite *suite.Suite) {
	points := g2.G2GenerateProjectivePoints(2)

	point1 := points[0]
	point2 := points[1]

	add := point1.Add(&point2)
	sub := add.Sub(&point2)

	suite.True(point1.ProjectiveEq(&sub))
}

func testG2ProjectiveFromAffine(suite *suite.Suite) {
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
	suite.Equal(expected, projectivePoint)
}

type G2CurveTestSuite struct {
	suite.Suite
}

func (s *G2CurveTestSuite) TestG2Curve() {
	s.Run("TestG2AffineZero", testWrapper(&s.Suite, testG2AffineZero))
	s.Run("TestG2AffineFromLimbs", testWrapper(&s.Suite, testG2AffineFromLimbs))
	s.Run("TestG2AffineToProjective", testWrapper(&s.Suite, testG2AffineToProjective))
	s.Run("TestG2ProjectiveZero", testWrapper(&s.Suite, testG2ProjectiveZero))
	s.Run("TestG2ProjectiveFromLimbs", testWrapper(&s.Suite, testG2ProjectiveFromLimbs))
	s.Run("TestG2ProjectiveFromAffine", testWrapper(&s.Suite, testG2ProjectiveFromAffine))
	s.Run("TestG2ProjectiveArithmetic", testWrapper(&s.Suite, testG2ProjectiveArithmetic))
}

func TestSuiteG2Curve(t *testing.T) {
	suite.Run(t, new(G2CurveTestSuite))
}
