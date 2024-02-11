package internal

import (
	"testing"
	"github.com/stretchr/testify/assert"
)

func TestFromLimbs(t *testing.T) {
	emptyField := Field{}
	randLimbs := [BASE_LIMBS]uint32{1,2,3,4,5,6,7,8}
	emptyField.FromLimbs(randLimbs[:])
	assert.ElementsMatch(t, randLimbs, emptyField.limbs, "Limbs do not match; there was an issue with setting the Field's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.limbs)
}

func TestGetLimbs(t *testing.T) {
	emptyField := Field{}
	randLimbs := [BASE_LIMBS]uint32{1,2,3,4,5,6,7,8}
	emptyField.FromLimbs(randLimbs[:])

	assert.ElementsMatch(t, randLimbs, emptyField.GetLimbs(), "Limbs do not match; there was an issue with setting the Field's limbs")
}

func TestFieldOne(t *testing.T) {
	var emptyField Field
	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), []uint32{1,0,0,0,0,0,0,0}, "Empty field to field one did not work")

	randLimbs := [BASE_LIMBS]uint32{1,2,3,4,5,6,7,8}
	emptyField.FromLimbs(randLimbs[:])

	emptyField.One()
	assert.ElementsMatch(t, emptyField.GetLimbs(), []uint32{1,0,0,0,0,0,0,0}, "Field with limbs to field one did not work")
}

func TestFieldZero(t *testing.T) {
	var emptyField Field
	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), []uint32{0,0,0,0,0,0,0,0}, "Empty field to field zero failed")

	randLimbs := [BASE_LIMBS]uint32{1,2,3,4,5,6,7,8}
	emptyField.FromLimbs(randLimbs[:])

	emptyField.Zero()
	assert.ElementsMatch(t, emptyField.GetLimbs(), []uint32{0,0,0,0,0,0,0,0}, "Field with limbs to field zero failed")
}

func TestFieldSize(t *testing.T) {
	var emptyField Field
	randLimbs := [BASE_LIMBS]uint32{1,2,3,4,5,6,7,8}
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, len(randLimbs)*4, emptyField.Size(), "Size returned an incorrect value of bytes")
}

func TestFieldAsPointer(t *testing.T) {
	var emptyField Field
	randLimbs := [BASE_LIMBS]uint32{1,2,3,4,5,6,7,8}
	emptyField.FromLimbs(randLimbs[:])

	assert.Equal(t, randLimbs[0], *emptyField.AsPointer(), "AsPointer returned pointer to incorrect value")
}

func TestFromBytes(t *testing.T) {
	var emptyField Field
	randomBytes := []byte{1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4}

	emptyField.FromBytesLittleEndian(randomBytes)

	expected := []uint32{67305985,67305985,67305985,67305985,67305985,67305985,67305985,67305985}
	assert.ElementsMatch(t, emptyField.GetLimbs(), expected, "FromBytes returned incorrect values")
}

func TestToBytes(t *testing.T) {
	var emptyField Field
	randLimbs := []uint32{67305985,67305985,67305985,67305985,67305985,67305985,67305985,67305985}
	emptyField.FromLimbs(randLimbs)
	
	expected := []byte{1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4}
	assert.ElementsMatch(t, emptyField.ToBytesLittleEndian(), expected, "ToBytes returned incorrect values")
}