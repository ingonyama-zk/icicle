//go:build !icicle_exclude_all || g2

package tests

import (
	"testing"

	bw6_761 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761/g2"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

const (
	G2BASE_LIMBS = bw6_761.G2BASE_LIMBS
)

func testG2BaseFieldFromLimbs(suite *suite.Suite) {
	emptyField := bw6_761.G2BaseField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the G2BaseField's limbs")
	randLimbs[0] = 100
	suite.NotEqual(randLimbs, emptyField.GetLimbs())
}

func testG2BaseFieldGetLimbs(suite *suite.Suite) {
	emptyField := bw6_761.G2BaseField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the G2BaseField's limbs")
}

func testG2BaseFieldOne(suite *suite.Suite) {
	var emptyField bw6_761.G2BaseField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(G2BASE_LIMBS))
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "G2BaseField with limbs to field one did not work")
}

func testG2BaseFieldZero(suite *suite.Suite) {
	var emptyField bw6_761.G2BaseField
	emptyField.Zero()
	limbsZero := make([]uint32, G2BASE_LIMBS)
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "G2BaseField with limbs to field zero failed")
}

func testG2BaseFieldSize(suite *suite.Suite) {
	var emptyField bw6_761.G2BaseField
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func testG2BaseFieldAsPointer(suite *suite.Suite) {
	var emptyField bw6_761.G2BaseField
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func testG2BaseFieldFromBytes(suite *suite.Suite) {
	var emptyField bw6_761.G2BaseField
	bytes, expected := test_helpers.GenerateBytesArray(int(G2BASE_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	suite.ElementsMatch(emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func testG2BaseFieldToBytes(suite *suite.Suite) {
	var emptyField bw6_761.G2BaseField
	expected, limbs := test_helpers.GenerateBytesArray(int(G2BASE_LIMBS))
	emptyField.FromLimbs(limbs)

	suite.ElementsMatch(emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

type G2BaseFieldTestSuite struct {
	suite.Suite
}

func (s *G2BaseFieldTestSuite) TestG2BaseField() {
	s.Run("TestG2BaseFieldFromLimbs", test_helpers.TestWrapper(&s.Suite, testG2BaseFieldFromLimbs))
	s.Run("TestG2BaseFieldGetLimbs", test_helpers.TestWrapper(&s.Suite, testG2BaseFieldGetLimbs))
	s.Run("TestG2BaseFieldOne", test_helpers.TestWrapper(&s.Suite, testG2BaseFieldOne))
	s.Run("TestG2BaseFieldZero", test_helpers.TestWrapper(&s.Suite, testG2BaseFieldZero))
	s.Run("TestG2BaseFieldSize", test_helpers.TestWrapper(&s.Suite, testG2BaseFieldSize))
	s.Run("TestG2BaseFieldAsPointer", test_helpers.TestWrapper(&s.Suite, testG2BaseFieldAsPointer))
	s.Run("TestG2BaseFieldFromBytes", test_helpers.TestWrapper(&s.Suite, testG2BaseFieldFromBytes))
	s.Run("TestG2BaseFieldToBytes", test_helpers.TestWrapper(&s.Suite, testG2BaseFieldToBytes))

}

func TestSuiteG2BaseField(t *testing.T) {
	suite.Run(t, new(G2BaseFieldTestSuite))
}
