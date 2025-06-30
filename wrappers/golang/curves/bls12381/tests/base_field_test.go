package tests

import (
	"testing"

	bls12_381 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

const (
	BASE_LIMBS = bls12_381.BASE_LIMBS
)

func testBaseFieldFromLimbs(suite *suite.Suite) {
	emptyField := bls12_381.BaseField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the BaseField's limbs")
	randLimbs[0] = 100
	suite.NotEqual(randLimbs, emptyField.GetLimbs())
}

func testBaseFieldGetLimbs(suite *suite.Suite) {
	emptyField := bls12_381.BaseField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the BaseField's limbs")
}

func testBaseFieldOne(suite *suite.Suite) {
	var emptyField bls12_381.BaseField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(BASE_LIMBS))
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "BaseField with limbs to field one did not work")
}

func testBaseFieldZero(suite *suite.Suite) {
	var emptyField bls12_381.BaseField
	emptyField.Zero()
	limbsZero := make([]uint32, BASE_LIMBS)
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "BaseField with limbs to field zero failed")
}

func testBaseFieldSize(suite *suite.Suite) {
	var emptyField bls12_381.BaseField
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func testBaseFieldAsPointer(suite *suite.Suite) {
	var emptyField bls12_381.BaseField
	randLimbs := test_helpers.GenerateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func testBaseFieldFromBytes(suite *suite.Suite) {
	var emptyField bls12_381.BaseField
	bytes, expected := test_helpers.GenerateBytesArray(int(BASE_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	suite.ElementsMatch(emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func testBaseFieldToBytes(suite *suite.Suite) {
	var emptyField bls12_381.BaseField
	expected, limbs := test_helpers.GenerateBytesArray(int(BASE_LIMBS))
	emptyField.FromLimbs(limbs)

	suite.ElementsMatch(emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

type BaseFieldTestSuite struct {
	suite.Suite
}

func (s *BaseFieldTestSuite) TestBaseField() {
	s.Run("TestBaseFieldFromLimbs", testWrapper(&s.Suite, testBaseFieldFromLimbs))
	s.Run("TestBaseFieldGetLimbs", testWrapper(&s.Suite, testBaseFieldGetLimbs))
	s.Run("TestBaseFieldOne", testWrapper(&s.Suite, testBaseFieldOne))
	s.Run("TestBaseFieldZero", testWrapper(&s.Suite, testBaseFieldZero))
	s.Run("TestBaseFieldSize", testWrapper(&s.Suite, testBaseFieldSize))
	s.Run("TestBaseFieldAsPointer", testWrapper(&s.Suite, testBaseFieldAsPointer))
	s.Run("TestBaseFieldFromBytes", testWrapper(&s.Suite, testBaseFieldFromBytes))
	s.Run("TestBaseFieldToBytes", testWrapper(&s.Suite, testBaseFieldToBytes))

}

func TestSuiteBaseField(t *testing.T) {
	suite.Run(t, new(BaseFieldTestSuite))
}
