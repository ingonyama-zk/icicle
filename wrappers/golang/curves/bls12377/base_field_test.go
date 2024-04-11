package bls12377

import (
	"testing"
  
	"github.com/stretchr/testify/assert"
)

func TestBaseFieldFromLimbs(t *testing.T) {
	emptyField := BaseField{}
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.limbs, "Limbs do not match; there was an issue with setting the BaseField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.limbs)
}

func TestBaseFieldGetLimbs(t *testing.T) {
	emptyField := BaseField{}
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the BaseField's limbs")
}

func TestBaseFieldOne(t *testing.T) {
	var emptyField BaseField
	emptyField.One()
	limbOne := generateLimbOne(int(BASE_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "BaseField with limbs to field one did not work")
}

func TestBaseFieldZero(t *testing.T) {
	var emptyField BaseField
	emptyField.Zero()
	limbsZero := make([]uint32, BASE_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "BaseField with limbs to field zero failed")
}

func TestBaseFieldSize(t *testing.T) {
	var emptyField BaseField
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestBaseFieldAsPointer(t *testing.T) {
	var emptyField BaseField
	randLimbs := generateRandomLimb(int(BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestBaseFieldFromBytes(t *testing.T) {
	var emptyField BaseField
	bytes, expected := generateBytesArray(int(BASE_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestBaseFieldToBytes(t *testing.T) {
	var emptyField BaseField
	expected, limbs := generateBytesArray(int(BASE_LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}
