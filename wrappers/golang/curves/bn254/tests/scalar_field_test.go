package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bn254 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

const (
	SCALAR_LIMBS = bn254.SCALAR_LIMBS
)

func testScalarFieldFromLimbs(suite *suite.Suite) {
	emptyField := bn254.ScalarField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
	randLimbs[0] = 100
	suite.NotEqual(randLimbs, emptyField.GetLimbs())
}

func testScalarFieldGetLimbs(suite *suite.Suite) {
	emptyField := bn254.ScalarField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
}

func testScalarFieldOne(suite *suite.Suite) {
	var emptyField bn254.ScalarField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(SCALAR_LIMBS))
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "ScalarField with limbs to field one did not work")
}

func testScalarFieldZero(suite *suite.Suite) {
	var emptyField bn254.ScalarField
	emptyField.Zero()
	limbsZero := make([]uint32, SCALAR_LIMBS)
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "ScalarField with limbs to field zero failed")
}

func testScalarFieldSize(suite *suite.Suite) {
	var emptyField bn254.ScalarField
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func testScalarFieldAsPointer(suite *suite.Suite) {
	var emptyField bn254.ScalarField
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func testScalarFieldFromBytes(suite *suite.Suite) {
	var emptyField bn254.ScalarField
	bytes, expected := test_helpers.GenerateBytesArray(int(SCALAR_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	suite.ElementsMatch(emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func testScalarFieldToBytes(suite *suite.Suite) {
	var emptyField bn254.ScalarField
	expected, limbs := test_helpers.GenerateBytesArray(int(SCALAR_LIMBS))
	emptyField.FromLimbs(limbs)

	suite.ElementsMatch(emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func testBn254GenerateScalars(suite *suite.Suite) {
	const numScalars = 8
	scalars := bn254.GenerateScalars(numScalars)

	suite.Implements((*core.HostOrDeviceSlice)(nil), &scalars)

	suite.Equal(numScalars, scalars.Len())
	zeroScalar := bn254.ScalarField{}
	suite.NotContains(scalars, zeroScalar)
}

func testBn254MongtomeryConversion(suite *suite.Suite) {
	size := 1 << 20
	scalars := bn254.GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	bn254.ToMontgomery(deviceScalars)

	scalarsMontHost := make(core.HostSlice[bn254.ScalarField], size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	suite.NotEqual(scalars, scalarsMontHost)

	bn254.FromMontgomery(deviceScalars)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	suite.Equal(scalars, scalarsMontHost)
}

type ScalarFieldTestSuite struct {
	suite.Suite
}

func (s *ScalarFieldTestSuite) TestScalarField() {
	s.Run("TestScalarFieldFromLimbs", testWrapper(&s.Suite, testScalarFieldFromLimbs))
	s.Run("TestScalarFieldGetLimbs", testWrapper(&s.Suite, testScalarFieldGetLimbs))
	s.Run("TestScalarFieldOne", testWrapper(&s.Suite, testScalarFieldOne))
	s.Run("TestScalarFieldZero", testWrapper(&s.Suite, testScalarFieldZero))
	s.Run("TestScalarFieldSize", testWrapper(&s.Suite, testScalarFieldSize))
	s.Run("TestScalarFieldAsPointer", testWrapper(&s.Suite, testScalarFieldAsPointer))
	s.Run("TestScalarFieldFromBytes", testWrapper(&s.Suite, testScalarFieldFromBytes))
	s.Run("TestScalarFieldToBytes", testWrapper(&s.Suite, testScalarFieldToBytes))
	s.Run("TestBn254GenerateScalars", testWrapper(&s.Suite, testBn254GenerateScalars))
	s.Run("TestBn254MongtomeryConversion", testWrapper(&s.Suite, testBn254MongtomeryConversion))
}

func TestSuiteScalarField(t *testing.T) {
	suite.Run(t, new(ScalarFieldTestSuite))
}
