package tests

import (
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	babybear_extension "github.com/ingonyama-zk/icicle/v2/wrappers/golang/fields/babybear/extension"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

const (
	EXTENSION_LIMBS = babybear_extension.EXTENSION_LIMBS
)

func TestExtensionFieldFromLimbs(t *testing.T) {
	emptyField := babybear_extension.ExtensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ExtensionField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.GetLimbs())
}

func TestExtensionFieldGetLimbs(t *testing.T) {
	emptyField := babybear_extension.ExtensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ExtensionField's limbs")
}

func TestExtensionFieldOne(t *testing.T) {
	var emptyField babybear_extension.ExtensionField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(EXTENSION_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "ExtensionField with limbs to field one did not work")
}

func TestExtensionFieldZero(t *testing.T) {
	var emptyField babybear_extension.ExtensionField
	emptyField.Zero()
	limbsZero := make([]uint32, EXTENSION_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "ExtensionField with limbs to field zero failed")
}

func TestExtensionFieldSize(t *testing.T) {
	var emptyField babybear_extension.ExtensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestExtensionFieldAsPointer(t *testing.T) {
	var emptyField babybear_extension.ExtensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestExtensionFieldFromBytes(t *testing.T) {
	var emptyField babybear_extension.ExtensionField
	bytes, expected := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestExtensionFieldToBytes(t *testing.T) {
	var emptyField babybear_extension.ExtensionField
	expected, limbs := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func TestBabybear_extensionGenerateScalars(t *testing.T) {
	const numScalars = 8
	scalars := babybear_extension.GenerateScalars(numScalars)

	assert.Implements(t, (*core.HostOrDeviceSlice)(nil), &scalars)

	assert.Equal(t, numScalars, scalars.Len())
	zeroScalar := babybear_extension.ExtensionField{}
	assert.NotContains(t, scalars, zeroScalar)
}

func TestBabybear_extensionMongtomeryConversion(t *testing.T) {
	size := 1 << 15
	scalars := babybear_extension.GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	babybear_extension.ToMontgomery(&deviceScalars)

	scalarsMontHost := babybear_extension.GenerateScalars(size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	assert.NotEqual(t, scalars, scalarsMontHost)

	babybear_extension.FromMontgomery(&deviceScalars)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	assert.Equal(t, scalars, scalarsMontHost)
}
