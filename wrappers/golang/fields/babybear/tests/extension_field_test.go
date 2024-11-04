package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear_extension "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/extension"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

const (
	EXTENSION_LIMBS = babybear_extension.EXTENSION_LIMBS
)

func testExtensionFieldFromLimbs(suite *suite.Suite) {
	emptyField := babybear_extension.ExtensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ExtensionField's limbs")
	randLimbs[0] = 100
	suite.NotEqual(randLimbs, emptyField.GetLimbs())
}

func testExtensionFieldGetLimbs(suite *suite.Suite) {
	emptyField := babybear_extension.ExtensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.ElementsMatch(randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ExtensionField's limbs")
}

func testExtensionFieldOne(suite *suite.Suite) {
	var emptyField babybear_extension.ExtensionField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(EXTENSION_LIMBS))
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	suite.ElementsMatch(emptyField.GetLimbs(), limbOne, "ExtensionField with limbs to field one did not work")
}

func testExtensionFieldZero(suite *suite.Suite) {
	var emptyField babybear_extension.ExtensionField
	emptyField.Zero()
	limbsZero := make([]uint32, EXTENSION_LIMBS)
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	suite.ElementsMatch(emptyField.GetLimbs(), limbsZero, "ExtensionField with limbs to field zero failed")
}

func testExtensionFieldSize(suite *suite.Suite) {
	var emptyField babybear_extension.ExtensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func testExtensionFieldAsPointer(suite *suite.Suite) {
	var emptyField babybear_extension.ExtensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	suite.Equal(randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func testExtensionFieldFromBytes(suite *suite.Suite) {
	var emptyField babybear_extension.ExtensionField
	bytes, expected := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	suite.ElementsMatch(emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func testExtensionFieldToBytes(suite *suite.Suite) {
	var emptyField babybear_extension.ExtensionField
	expected, limbs := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(limbs)

	suite.ElementsMatch(emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func testBabybear_extensionGenerateScalars(suite *suite.Suite) {
	const numScalars = 8
	scalars := babybear_extension.GenerateScalars(numScalars)

	suite.Implements((*core.HostOrDeviceSlice)(nil), &scalars)

	suite.Equal(numScalars, scalars.Len())
	zeroScalar := babybear_extension.ExtensionField{}
	suite.NotContains(scalars, zeroScalar)
}

func testBabybear_extensionMongtomeryConversion(suite *suite.Suite) {
	size := 1 << 20
	scalars := babybear_extension.GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	babybear_extension.ToMontgomery(deviceScalars)

	scalarsMontHost := make(core.HostSlice[babybear_extension.ExtensionField], size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	suite.NotEqual(scalars, scalarsMontHost)

	babybear_extension.FromMontgomery(deviceScalars)

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
	s.Run("TestBabybear_extensionGenerateScalars", testWrapper(&s.Suite, testBabybear_extensionGenerateScalars))
	s.Run("TestBabybear_extensionMongtomeryConversion", testWrapper(&s.Suite, testBabybear_extensionMongtomeryConversion))
}

func TestSuiteExtensionField(t *testing.T) {
	suite.Run(t, new(ExtensionFieldTestSuite))
}
