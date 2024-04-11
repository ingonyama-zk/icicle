package babybear

import (
  "github.com/ingonyama-zk/icicle/wrappers/golang/core"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestFieldFromLimbs(t *testing.T) {
	emptyField := Field{}
	randLimbs := generateRandomLimb(int(LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.limbs, "Limbs do not match; there was an issue with setting the Field's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.limbs)
}

func TestFieldGetLimbs(t *testing.T) {
	emptyField := Field{}
	randLimbs := generateRandomLimb(int(LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the Field's limbs")
}

func TestFieldOne(t *testing.T) {
	var emptyField Field
	emptyField.One()
	limbOne := generateLimbOne(int(LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := generateRandomLimb(int(LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Field with limbs to field one did not work")
}

func TestFieldZero(t *testing.T) {
	var emptyField Field
	emptyField.Zero()
	limbsZero := make([]uint32, LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := generateRandomLimb(int(LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Field with limbs to field zero failed")
}

func TestFieldSize(t *testing.T) {
	var emptyField Field
	randLimbs := generateRandomLimb(int(LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestFieldAsPointer(t *testing.T) {
	var emptyField Field
	randLimbs := generateRandomLimb(int(LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestFieldFromBytes(t *testing.T) {
	var emptyField Field
	bytes, expected := generateBytesArray(int(LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestFieldToBytes(t *testing.T) {
	var emptyField Field
	expected, limbs := generateBytesArray(int(LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func TestGenerateScalars(t *testing.T) {
	const numScalars = 8
	scalars := GenerateScalars(numScalars)

	assert.Implements(t, (*core.HostOrDeviceSlice)(nil), &scalars)

	assert.Equal(t, numScalars, scalars.Len())
	zeroScalar := ScalarField{}
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
