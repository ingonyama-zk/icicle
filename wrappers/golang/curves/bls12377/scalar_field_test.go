package bls12377

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestScalarFieldFromLimbs(t *testing.T) {
	emptyField := ScalarField{}
	randLimbs := generateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.limbs, "Limbs do not match; there was an issue with setting the ScalarField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.limbs)
}

func TestScalarFieldGetLimbs(t *testing.T) {
	emptyField := ScalarField{}
	randLimbs := generateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
}

func TestScalarFieldOne(t *testing.T) {
	var emptyField ScalarField
	emptyField.One()
	limbOne := generateLimbOne(int(SCALAR_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := generateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "ScalarField with limbs to field one did not work")
}

func TestScalarFieldZero(t *testing.T) {
	var emptyField ScalarField
	emptyField.Zero()
	limbsZero := make([]uint64, SCALAR_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := generateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "ScalarField with limbs to field zero failed")
}

func TestScalarFieldSize(t *testing.T) {
	var emptyField ScalarField
	randLimbs := generateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*8, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestScalarFieldAsPointer(t *testing.T) {
	var emptyField ScalarField
	randLimbs := generateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestScalarFieldFromBytes(t *testing.T) {
	var emptyField ScalarField
	bytes, expected := generateBytesArray(int(SCALAR_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestScalarFieldToBytes(t *testing.T) {
	var emptyField ScalarField
	expected, limbs := generateBytesArray(int(SCALAR_LIMBS))
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
