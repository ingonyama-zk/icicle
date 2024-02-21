package internal

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestFromLimbs(t *testing.T) {
	emptyMockField := MockField{}
	randLimbs := [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8}
	emptyMockField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyMockField.limbs, "Limbs do not match; there was an issue with setting the MockField's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyMockField.limbs)
}

func TestGetLimbs(t *testing.T) {
	emptyMockField := MockField{}
	randLimbs := [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8}
	emptyMockField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyMockField.GetLimbs(), "Limbs do not match; there was an issue with setting the MockField's limbs")
}

func TestMockFieldOne(t *testing.T) {
	var emptyMockField MockField
	emptyMockField.One()
	assert.ElementsMatch(t, emptyMockField.GetLimbs(), []uint32{1, 0, 0, 0, 0, 0, 0, 0}, "Empty field to field one did not work")

	randLimbs := [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8}
	emptyMockField.FromLimbs(randLimbs[:])

	emptyMockField.One()
	assert.ElementsMatch(t, emptyMockField.GetLimbs(), []uint32{1, 0, 0, 0, 0, 0, 0, 0}, "MockField with limbs to field one did not work")
}

func TestMockFieldZero(t *testing.T) {
	var emptyMockField MockField
	emptyMockField.Zero()
	assert.ElementsMatch(t, emptyMockField.GetLimbs(), []uint32{0, 0, 0, 0, 0, 0, 0, 0}, "Empty field to field zero failed")

	randLimbs := [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8}
	emptyMockField.FromLimbs(randLimbs[:])

	emptyMockField.Zero()
	assert.ElementsMatch(t, emptyMockField.GetLimbs(), []uint32{0, 0, 0, 0, 0, 0, 0, 0}, "MockField with limbs to field zero failed")
}

func TestMockFieldSize(t *testing.T) {
	var emptyMockField MockField
	randLimbs := [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8}
	emptyMockField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyMockField.Size(), "Size returned an incorrect value of bytes")
}

func TestMockFieldAsPointer(t *testing.T) {
	var emptyMockField MockField
	randLimbs := [BASE_LIMBS]uint32{1, 2, 3, 4, 5, 6, 7, 8}
	emptyMockField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyMockField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestFromBytes(t *testing.T) {
	var emptyMockField MockField
	randomBytes := []byte{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}

	emptyMockField.FromBytesLittleEndian(randomBytes)

	expected := []uint32{67305985, 67305985, 67305985, 67305985, 67305985, 67305985, 67305985, 67305985}
	assert.ElementsMatch(t, emptyMockField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestToBytes(t *testing.T) {
	var emptyMockField MockField
	randLimbs := []uint32{67305985, 67305985, 67305985, 67305985, 67305985, 67305985, 67305985, 67305985}
	emptyMockField.FromLimbs(randLimbs)

	expected := []byte{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}
	assert.ElementsMatch(t, emptyMockField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}
