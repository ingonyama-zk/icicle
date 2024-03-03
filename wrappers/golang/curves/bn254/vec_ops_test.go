package bn254

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	"github.com/stretchr/testify/assert"
)

func TestVecOps(t *testing.T) {
	testSize := 1 << 14

	a := GenerateScalars(testSize)
	b := GenerateScalars(testSize)
	var scalar ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[ScalarField], testSize)
	out2 := make(core.HostSlice[ScalarField], testSize)
	out3 := make(core.HostSlice[ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	VecOp(a, b, out, cfg, core.Add)
	VecOp(out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	VecOp(a, ones, out3, cfg, core.Mul)

	assert.Equal(t, a, out3)
}
