package core

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestVecOpsDefaultConfig(t *testing.T) {
	ctx, _ := cuda_runtime.GetDefaultDeviceContext()
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
