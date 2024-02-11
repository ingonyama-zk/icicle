package bn254

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFromLimbs(t *testing.T) {
	emptyField := BaseField{}
	randLimbs := []uint32{1,2,3,4,5,6,7,8}
	emptyField.FromLimbs(randLimbs)
	assert.ElementsMatch(t, randLimbs, emptyField.limbs, "Limbs do not match; there was an issue with setting the Field's limbs")
	randLimbs[0] = 100
	assert.NotEqual(t, randLimbs, emptyField.limbs)
}