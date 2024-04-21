package tests

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	babybearExtension "github.com/ingonyama-zk/icicle/wrappers/golang/fields/babybear/extension"
	"github.com/ingonyama-zk/icicle/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

const (
	EXTENSION_LIMBS = babybearExtension.EXTENSION_LIMBS
)

func TestExtensionFieldFromLimbs(t *testing.T) {
	emptyField := babybearExtension.ExtensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ExtensionField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.GetLimbs())
}

func TestExtensionFieldGetLimbs(t *testing.T) {
	emptyField := babybearExtension.ExtensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ExtensionField's limbs")
}

func TestExtensionFieldOne(t *testing.T) {
	var emptyField babybearExtension.ExtensionField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(EXTENSION_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "ExtensionField with limbs to field one did not work")
}

func TestExtensionFieldZero(t *testing.T) {
	var emptyField babybearExtension.ExtensionField
	emptyField.Zero()
	limbsZero := make([]uint32, EXTENSION_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "ExtensionField with limbs to field zero failed")
}

func TestExtensionFieldSize(t *testing.T) {
	var emptyField babybearExtension.ExtensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestExtensionFieldAsPointer(t *testing.T) {
	var emptyField babybearExtension.ExtensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestExtensionFieldFromBytes(t *testing.T) {
	var emptyField babybearExtension.ExtensionField
	bytes, expected := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestExtensionFieldToBytes(t *testing.T) {
	var emptyField babybearExtension.ExtensionField
	expected, limbs := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func TestBabybearExtensionGenerateScalars(t *testing.T) {
	const numScalars = 8
	scalars := babybearExtension.GenerateScalars(numScalars)

	assert.Implements(t, (*core.HostOrDeviceSlice)(nil), &scalars)

	assert.Equal(t, numScalars, scalars.Len())
	zeroScalar := babybearExtension.ExtensionField{}
	assert.NotContains(t, scalars, zeroScalar)
}

func TestBabybearExtensionMongtomeryConversion(t *testing.T) {
	size := 1 << 15
	scalars := babybearExtension.GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	babybearExtension.ToMontgomery(&deviceScalars)

	scalarsMontHost := babybearExtension.GenerateScalars(size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	assert.NotEqual(t, scalars, scalarsMontHost)

	babybearExtension.FromMontgomery(&deviceScalars)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	assert.Equal(t, scalars, scalarsMontHost)
}
