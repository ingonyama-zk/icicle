package internal

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMockFieldFromLimbs(t *testing.T) {
	emptyField := MockField{}
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.limbs, "Limbs do not match; there was an issue with setting the MockField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.limbs)
}

func TestMockFieldGetLimbs(t *testing.T) {
	emptyField := MockField{}
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the MockField's limbs")
}

func TestMockFieldOne(t *testing.T) {
	var emptyField MockField
	emptyField.One()
	limbOne := generateLimbOne(int(BASE_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "MockField with limbs to field one did not work")
}

func TestMockFieldZero(t *testing.T) {
	var emptyField MockField
	emptyField.Zero()
	limbsZero := make([]uint64, BASE_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "MockField with limbs to field zero failed")
}

func TestMockFieldSize(t *testing.T) {
	var emptyField MockField
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*8, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestMockFieldAsPointer(t *testing.T) {
	var emptyField MockField
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestMockFieldFromBytes(t *testing.T) {
	var emptyField MockField
	bytes, expected := generateBytesArray(int(BASE_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestMockFieldToBytes(t *testing.T) {
	var emptyField MockField
	expected, limbs := generateBytesArray(int(BASE_LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}
