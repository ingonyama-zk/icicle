package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	grumpkin "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

const (
	SCALAR_LIMBS = grumpkin.SCALAR_LIMBS
)

func testScalarFieldFromLimbs(suite *suite.Suite) {
	emptyField := grumpkin.ScalarField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
	randLimbs[0] = 100
	suite.NotEqual(randLimbs, emptyField.GetLimbs())
}

func testScalarFieldGetLimbs(suite *suite.Suite) {
	emptyField := grumpkin.ScalarField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
}

func testScalarFieldOne(suite *suite.Suite) {
	var emptyField grumpkin.ScalarField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(SCALAR_LIMBS))
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "ScalarField with limbs to field one did not work")
}

func testScalarFieldZero(suite *suite.Suite) {
	var emptyField grumpkin.ScalarField
	emptyField.Zero()
	limbsZero := make([]uint32, SCALAR_LIMBS)
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "ScalarField with limbs to field zero failed")
}

func testScalarFieldSize(suite *suite.Suite) {
	var emptyField grumpkin.ScalarField
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func testScalarFieldAsPointer(suite *suite.Suite) {
	var emptyField grumpkin.ScalarField
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func testScalarFieldFromBytes(suite *suite.Suite) {
	var emptyField grumpkin.ScalarField
	bytes, expected := test_helpers.GenerateBytesArray(int(SCALAR_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	suite.ElementsMatch(emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func testScalarFieldToBytes(suite *suite.Suite) {
	var emptyField grumpkin.ScalarField
	expected, limbs := test_helpers.GenerateBytesArray(int(SCALAR_LIMBS))
	emptyField.FromLimbs(limbs)

	suite.ElementsMatch(emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func testGrumpkinGenerateScalars(suite *suite.Suite) {
	const numScalars = 8
	scalars := grumpkin.GenerateScalars(numScalars)

	suite.Implements((*core.HostOrDeviceSlice)(nil), &scalars)

	suite.Equal(numScalars, scalars.Len())
	zeroScalar := grumpkin.ScalarField{}
	suite.NotContains(scalars, zeroScalar)
}

func testScalarFieldArithmetic(suite *suite.Suite) {
	const size = 1 << 10

	scalarsA := grumpkin.GenerateScalars(size)
	scalarsB := grumpkin.GenerateScalars(size)

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
	expectedOne := grumpkin.GenerateScalars(1)[0]
	expectedOne.One()

	suite.Equal(expectedOne, one)
}

func testGrumpkinMongtomeryConversion(suite *suite.Suite) {
	size := 1 << 20
	scalars := grumpkin.GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	grumpkin.ToMontgomery(deviceScalars)

	scalarsMontHost := make(core.HostSlice[grumpkin.ScalarField], size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	suite.NotEqual(scalars, scalarsMontHost)

	grumpkin.FromMontgomery(deviceScalars)

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
	s.Run("TestScalarFieldArithmetic", testWrapper(&s.Suite, testScalarFieldArithmetic))
	s.Run("TestGrumpkinGenerateScalars", testWrapper(&s.Suite, testGrumpkinGenerateScalars))
	s.Run("TestGrumpkinMongtomeryConversion", testWrapper(&s.Suite, testGrumpkinMongtomeryConversion))
}

func TestSuiteScalarField(t *testing.T) {
	suite.Run(t, new(ScalarFieldTestSuite))
}
