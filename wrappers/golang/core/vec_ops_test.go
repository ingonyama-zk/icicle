package core

import (
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestVecOpsDefaultConfig(t *testing.T) {
	ctx, _ := cr.GetDefaultDeviceContext()
	expected := VecOpsConfig{
		ctx,   // Ctx
		false, // isAOnDevice
		false, // isBOnDevice
		false, // isResultOnDevice
		false, // IsResultMontgomeryForm
		false, // IsAsync
	}

	actual := DefaultVecOpsConfig()

	assert.Equal(t, expected, actual)
}
