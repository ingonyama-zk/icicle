package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
)

const (
	SCALAR_LIMBS = bls12_377.SCALAR_LIMBS
)

func TestScalarFieldFromLimbs(t *testing.T) {
	emptyField := bls12_377.ScalarField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.GetLimbs())
}

func TestScalarFieldGetLimbs(t *testing.T) {
	emptyField := bls12_377.ScalarField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the ScalarField's limbs")
}

func TestScalarFieldOne(t *testing.T) {
	var emptyField bls12_377.ScalarField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(SCALAR_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "ScalarField with limbs to field one did not work")
}

func TestScalarFieldZero(t *testing.T) {
	var emptyField bls12_377.ScalarField
	emptyField.Zero()
	limbsZero := make([]uint32, SCALAR_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "ScalarField with limbs to field zero failed")
}

func TestScalarFieldSize(t *testing.T) {
	var emptyField bls12_377.ScalarField
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestScalarFieldAsPointer(t *testing.T) {
	var emptyField bls12_377.ScalarField
	randLimbs := test_helpers.GenerateRandomLimb(int(SCALAR_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestScalarFieldFromBytes(t *testing.T) {
	var emptyField bls12_377.ScalarField
	bytes, expected := test_helpers.GenerateBytesArray(int(SCALAR_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestScalarFieldToBytes(t *testing.T) {
	var emptyField bls12_377.ScalarField
	expected, limbs := test_helpers.GenerateBytesArray(int(SCALAR_LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}

func TestBls12_377GenerateScalars(t *testing.T) {
	const numScalars = 8
	scalars := bls12_377.GenerateScalars(numScalars)

	assert.Implements(t, (*core.HostOrDeviceSlice)(nil), &scalars)

	assert.Equal(t, numScalars, scalars.Len())
	zeroScalar := bls12_377.ScalarField{}
	assert.NotContains(t, scalars, zeroScalar)
}

func TestBls12_377MongtomeryConversion(t *testing.T) {
	size := 1 << 15
	scalars := bls12_377.GenerateScalars(size)

	var deviceScalars core.DeviceSlice
	scalars.CopyToDevice(&deviceScalars, true)

	bls12_377.ToMontgomery(&deviceScalars)

	scalarsMontHost := bls12_377.GenerateScalars(size)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	assert.NotEqual(t, scalars, scalarsMontHost)

	bls12_377.FromMontgomery(&deviceScalars)

	scalarsMontHost.CopyFromDevice(&deviceScalars)
	assert.Equal(t, scalars, scalarsMontHost)
}
