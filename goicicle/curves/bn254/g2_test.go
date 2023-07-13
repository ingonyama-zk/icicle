package bn254

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestToGnarkJacG2(t *testing.T) {
	gnark, _ := randG2Jac()

	var pointAffine G2PointAffine
	pointAffine.FromGnarkJac(&gnark)
	pointProjective := pointAffine.ToProjective()
	backToGnark := pointProjective.ToGnarkJac()

	assert.True(t, gnark.Equal(backToGnark))
}
