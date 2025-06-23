package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	stark252 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/stark252"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

const (
	SCALAR_LIMBS = stark252.SCALAR_LIMBS
)

func testScalarFieldFromLimbs(suite *suite.Suite) {
	emptyField := stark252.ScalarField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
	randLimbs[0] = 100
	suite.NotEqual(randLimbs, emptyField.GetLimbs())
}

func testScalarFieldGetLimbs(suite *suite.Suite) {
	emptyField := stark252.ScalarField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
}

func testScalarFieldOne(suite *suite.Suite) {
	var emptyField stark252.ScalarField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(SCALAR_LIMBS))
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "ScalarField with limbs to field one did not work")
}

func testScalarFieldZero(suite *suite.Suite) {
	var emptyField stark252.ScalarField
	emptyField.Zero()
	limbsZero := make([]uint32, SCALAR_LIMBS)
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "ScalarField with limbs to field zero failed")
}

func testScalarFieldSize(suite *suite.Suite) {
	var emptyField stark252.ScalarField
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func testScalarFieldAsPointer(suite *suite.Suite) {
	var emptyField stark252.ScalarField
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func testScalarFieldFromBytes(suite *suite.Suite) {
	var emptyField stark252.ScalarField
	bytes, expected := test_helpers.GenerateBytesArray(int(SCALAR_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	suite.ElementsMatch(emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func testScalarFieldToBytes(suite *suite.Suite) {
	var emptyField stark252.ScalarField
	expected, limbs := test_helpers.GenerateBytesArray(int(SCALAR_LIMBS))
	emptyField.FromLimbs(limbs)

	suite.ElementsMatch(emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func testStark252GenerateScalars(suite *suite.Suite) {
	const numScalars = 8
	scalars := stark252.GenerateScalars(numScalars)

	suite.Implements((*core.HostOrDeviceSlice)(nil), &scalars)

	suite.Equal(numScalars, scalars.Len())
	zeroScalar := stark252.ScalarField{}
	suite.NotContains(scalars, zeroScalar)
}

func testScalarFieldArithmetic(suite *suite.Suite) {
	const size = 1 << 10

	scalarsA := stark252.GenerateScalars(size)
	scalarsB := stark252.GenerateScalars(size)

	for i := 0; i < size; i++ {
		result1 := scalarsA[i].Add(&scalarsB[i])
		result2 := result1.Sub(&scalarsB[i])

		suite.Equal(scalarsA[i], result2, "Addition and subtraction do not yield the original value")
	}

	scalarA := scalarsA[0]
	square := scalarA.Sqr()
	mul := scalarA.Mul(&scalarA)

	suite.Equal(square, mul, "Square and multiplication do not yield the same value")

	pow4 := scalarA.Pow(4)
	mulBySelf := mul.Mul(&mul)

	suite.Equal(pow4, mulBySelf, "Square and multiplication do not yield the same value")

	inv := scalarA.Inv()

	one := scalarA.Mul(&inv)
	expectedOne := stark252.GenerateScalars(1)[0]
	expectedOne.One()

	suite.Equal(expectedOne, one)
}

func testStark252MongtomeryConversion(suite *suite.Suite) {
	size := 1 << 20
	scalars := stark252.GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	stark252.ToMontgomery(deviceScalars)

	scalarsMontHost := make(core.HostSlice[stark252.ScalarField], size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	suite.NotEqual(scalars, scalarsMontHost)

	stark252.FromMontgomery(deviceScalars)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	suite.Equal(scalars, scalarsMontHost)
}

type ScalarFieldTestSuite struct {
	suite.Suite
}

func (s *ScalarFieldTestSuite) TestScalarField() {
	s.Run("TestScalarFieldFromLimbs", test_helpers.TestWrapper(&s.Suite, testScalarFieldFromLimbs))
	s.Run("TestScalarFieldGetLimbs", test_helpers.TestWrapper(&s.Suite, testScalarFieldGetLimbs))
	s.Run("TestScalarFieldOne", test_helpers.TestWrapper(&s.Suite, testScalarFieldOne))
	s.Run("TestScalarFieldZero", test_helpers.TestWrapper(&s.Suite, testScalarFieldZero))
	s.Run("TestScalarFieldSize", test_helpers.TestWrapper(&s.Suite, testScalarFieldSize))
	s.Run("TestScalarFieldAsPointer", test_helpers.TestWrapper(&s.Suite, testScalarFieldAsPointer))
	s.Run("TestScalarFieldFromBytes", test_helpers.TestWrapper(&s.Suite, testScalarFieldFromBytes))
	s.Run("TestScalarFieldToBytes", test_helpers.TestWrapper(&s.Suite, testScalarFieldToBytes))
	s.Run("TestScalarFieldArithmetic", test_helpers.TestWrapper(&s.Suite, testScalarFieldArithmetic))
	s.Run("TestStark252GenerateScalars", test_helpers.TestWrapper(&s.Suite, testStark252GenerateScalars))
	s.Run("TestStark252MongtomeryConversion", test_helpers.TestWrapper(&s.Suite, testStark252MongtomeryConversion))
}

func TestSuiteScalarField(t *testing.T) {
	suite.Run(t, new(ScalarFieldTestSuite))
}
