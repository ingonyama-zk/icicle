package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	koalabear_extension "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear/extension"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

const (
	EXTENSION_LIMBS = koalabear_extension.EXTENSION_LIMBS
)

func testExtensionFieldFromLimbs(suite *suite.Suite) {
	emptyField := koalabear_extension.ExtensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ExtensionField's limbs")
	randLimbs[0] = 100
	suite.NotEqual(randLimbs, emptyField.GetLimbs())
}

func testExtensionFieldGetLimbs(suite *suite.Suite) {
	emptyField := koalabear_extension.ExtensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ExtensionField's limbs")
}

func testExtensionFieldOne(suite *suite.Suite) {
	var emptyField koalabear_extension.ExtensionField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(EXTENSION_LIMBS))
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "ExtensionField with limbs to field one did not work")
}

func testExtensionFieldZero(suite *suite.Suite) {
	var emptyField koalabear_extension.ExtensionField
	emptyField.Zero()
	limbsZero := make([]uint32, EXTENSION_LIMBS)
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "ExtensionField with limbs to field zero failed")
}

func testExtensionFieldSize(suite *suite.Suite) {
	var emptyField koalabear_extension.ExtensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func testExtensionFieldAsPointer(suite *suite.Suite) {
	var emptyField koalabear_extension.ExtensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func testExtensionFieldFromBytes(suite *suite.Suite) {
	var emptyField koalabear_extension.ExtensionField
	bytes, expected := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	suite.ElementsMatch(emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func testExtensionFieldToBytes(suite *suite.Suite) {
	var emptyField koalabear_extension.ExtensionField
	expected, limbs := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(limbs)

	suite.ElementsMatch(emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func testKoalabear_extensionGenerateScalars(suite *suite.Suite) {
	const numScalars = 8
	scalars := koalabear_extension.GenerateScalars(numScalars)

	suite.Implements((*core.HostOrDeviceSlice)(nil), &scalars)

	suite.Equal(numScalars, scalars.Len())
	zeroScalar := koalabear_extension.ExtensionField{}
	suite.NotContains(scalars, zeroScalar)
}

func testExtensionFieldArithmetic(suite *suite.Suite) {
	const size = 1 << 10

	scalarsA := koalabear_extension.GenerateScalars(size)
	scalarsB := koalabear_extension.GenerateScalars(size)

	for i := 0; i < size; i++ {
		result1 := scalarsA[i].Add(&scalarsB[i])
		result2 := result1.Sub(&scalarsB[i])

		suite.Equal(scalarsA[i], result2, "Addition and subtraction do not yield the original value")
	}

	scalarA := scalarsA[0]
	square := scalarA.Sqr()
	mul := scalarA.Mul(&scalarA)

	suite.Equal(square, mul, "Square and multiplication do not yield the same value")

	inv := scalarA.Inv()

	one := scalarA.Mul(&inv)
	expectedOne := koalabear_extension.GenerateScalars(1)[0]
	expectedOne.One()

	suite.Equal(expectedOne, one)
}

func testKoalabear_extensionMongtomeryConversion(suite *suite.Suite) {
	size := 1 << 20
	scalars := koalabear_extension.GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	koalabear_extension.ToMontgomery(deviceScalars)

	scalarsMontHost := make(core.HostSlice[koalabear_extension.ExtensionField], size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	suite.NotEqual(scalars, scalarsMontHost)

	koalabear_extension.FromMontgomery(deviceScalars)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	suite.Equal(scalars, scalarsMontHost)
}

type ExtensionFieldTestSuite struct {
	suite.Suite
}

func (s *ExtensionFieldTestSuite) TestExtensionField() {
	s.Run("TestExtensionFieldFromLimbs", testWrapper(&s.Suite, testExtensionFieldFromLimbs))
	s.Run("TestExtensionFieldGetLimbs", testWrapper(&s.Suite, testExtensionFieldGetLimbs))
	s.Run("TestExtensionFieldOne", testWrapper(&s.Suite, testExtensionFieldOne))
	s.Run("TestExtensionFieldZero", testWrapper(&s.Suite, testExtensionFieldZero))
	s.Run("TestExtensionFieldSize", testWrapper(&s.Suite, testExtensionFieldSize))
	s.Run("TestExtensionFieldAsPointer", testWrapper(&s.Suite, testExtensionFieldAsPointer))
	s.Run("TestExtensionFieldFromBytes", testWrapper(&s.Suite, testExtensionFieldFromBytes))
	s.Run("TestExtensionFieldToBytes", testWrapper(&s.Suite, testExtensionFieldToBytes))
	s.Run("TestExtensionFieldArithmetic", testWrapper(&s.Suite, testExtensionFieldArithmetic))
	s.Run("TestKoalabear_extensionGenerateScalars", testWrapper(&s.Suite, testKoalabear_extensionGenerateScalars))
	s.Run("TestKoalabear_extensionMongtomeryConversion", testWrapper(&s.Suite, testKoalabear_extensionMongtomeryConversion))
}

func TestSuiteExtensionField(t *testing.T) {
	suite.Run(t, new(ExtensionFieldTestSuite))
}
