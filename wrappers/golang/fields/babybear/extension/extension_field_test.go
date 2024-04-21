package extension

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestextensionFieldFromLimbs(t *testing.T) {
	emptyField := extensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.limbs, "Limbs do not match; there was an issue with setting the extensionField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.limbs)
}

func TestextensionFieldGetLimbs(t *testing.T) {
	emptyField := extensionField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the extensionField's limbs")
}

func TestextensionFieldOne(t *testing.T) {
	var emptyField extensionField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(EXTENSION_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "extensionField with limbs to field one did not work")
}

func TestextensionFieldZero(t *testing.T) {
	var emptyField extensionField
	emptyField.Zero()
	limbsZero := make([]uint32, EXTENSION_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "extensionField with limbs to field zero failed")
}

func TestextensionFieldSize(t *testing.T) {
	var emptyField extensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestextensionFieldAsPointer(t *testing.T) {
	var emptyField extensionField
	randLimbs := test_helpers.GenerateRandomLimb(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestextensionFieldFromBytes(t *testing.T) {
	var emptyField extensionField
	bytes, expected := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestextensionFieldToBytes(t *testing.T) {
	var emptyField extensionField
	expected, limbs := test_helpers.GenerateBytesArray(int(EXTENSION_LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func TestGenerateScalars(t *testing.T) {
	const numScalars = 8
	scalars := GenerateScalars(numScalars)

	assert.Implements(t, (*core.HostOrDeviceSlice)(nil), &scalars)

	assert.Equal(t, numScalars, scalars.Len())
	zeroScalar := extensionField{}
	assert.NotContains(t, scalars, zeroScalar)
}

func TestMongtomeryConversion(t *testing.T) {
	size := 1 << 15
	scalars := GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	ToMontgomery(&deviceScalars)

	scalarsMontHost := GenerateScalars(size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	assert.NotEqual(t, scalars, scalarsMontHost)

	FromMontgomery(&deviceScalars)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	assert.Equal(t, scalars, scalarsMontHost)
}
