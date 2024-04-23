package tests

import (
	bn254 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254/g2"
	"github.com/ingonyama-zk/icicle/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
	"testing"
)

const (
	G2BASE_LIMBS = bn254.G2BASE_LIMBS
)

func TestG2BaseFieldFromLimbs(t *testing.T) {
	emptyField := bn254.G2BaseField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the G2BaseField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.GetLimbs())
}

func TestG2BaseFieldGetLimbs(t *testing.T) {
	emptyField := bn254.G2BaseField{}
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the G2BaseField's limbs")
}

func TestG2BaseFieldOne(t *testing.T) {
	var emptyField bn254.G2BaseField
	emptyField.One()
	limbOne := test_helpers.GenerateLimbOne(int(G2BASE_LIMBS))
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "Empty field to field one did not work")

	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbOne, "G2BaseField with limbs to field one did not work")
}

func TestG2BaseFieldZero(t *testing.T) {
	var emptyField bn254.G2BaseField
	emptyField.Zero()
	limbsZero := make([]uint32, G2BASE_LIMBS)
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "Empty field to field zero failed")

	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), limbsZero, "G2BaseField with limbs to field zero failed")
}

func TestG2BaseFieldSize(t *testing.T) {
	var emptyField bn254.G2BaseField
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestG2BaseFieldAsPointer(t *testing.T) {
	var emptyField bn254.G2BaseField
	randLimbs := test_helpers.GenerateRandomLimb(int(G2BASE_LIMBS))
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestG2BaseFieldFromBytes(t *testing.T) {
	var emptyField bn254.G2BaseField
	bytes, expected := test_helpers.GenerateBytesArray(int(G2BASE_LIMBS))

	emptyField.FromBytesLittleEndian(bytes)

	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestG2BaseFieldToBytes(t *testing.T) {
	var emptyField bn254.G2BaseField
	expected, limbs := test_helpers.GenerateBytesArray(int(G2BASE_LIMBS))
	emptyField.FromLimbs(limbs)

	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}
