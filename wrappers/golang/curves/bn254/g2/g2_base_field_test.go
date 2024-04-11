package g2

import (
	"testing"
  
	"github.com/stretchr/testify/assert"
)

func TestG2BaseFieldFromLimbs(t *testing.T) {
	emptyField := G2BaseField{}
	randLimbs := generateRandomLimb(int(G2_BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.limbs, "Limbs do not match; there was an issue with setting the G2BaseField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.limbs)
}

func TestG2BaseFieldGetLimbs(t *testing.T) {
	emptyField := G2BaseField{}
	randLimbs := generateRandomLimb(int(G2_BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the G2BaseField's limbs")
}

func TestG2BaseFieldOne(t *testing.T) {
	var emptyField G2BaseField
	emptyField.One()
	limbOne := generateLimbOne(int(G2_BASE_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := generateRandomLimb(int(G2_BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "G2BaseField with limbs to field one did not work")
}

func TestG2BaseFieldZero(t *testing.T) {
	var emptyField G2BaseField
	emptyField.Zero()
	limbsZero := make([]uint64, G2_BASE_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := generateRandomLimb(int(G2_BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "G2BaseField with limbs to field zero failed")
}

func TestG2BaseFieldSize(t *testing.T) {
	var emptyField G2BaseField
	randLimbs := generateRandomLimb(int(G2_BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*8, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestG2BaseFieldAsPointer(t *testing.T) {
	var emptyField G2BaseField
	randLimbs := generateRandomLimb(int(G2_BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestG2BaseFieldFromBytes(t *testing.T) {
	var emptyField G2BaseField
	bytes, expected := generateBytesArray(int(G2_BASE_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestG2BaseFieldToBytes(t *testing.T) {
	var emptyField G2BaseField
	expected, limbs := generateBytesArray(int(G2_BASE_LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}
