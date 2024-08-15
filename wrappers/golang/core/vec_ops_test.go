package core

import (
	"testing"

	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	"github.com/stretchr/testify/assert"
)

func TestVecOpsDefaultConfig(t *testing.T) {
	ctx, _ := cr.GetDefaultDeviceContext()
	expected := VecOpsConfig{
		ctx,   // Ctx
		false, // isAOnDevice
		false, // isBOnDevice
		false, // isResultOnDevice
		false, // IsAsync
		false, // IsInputInMontgomeryForm
		false, // IsResultInMontgomeryForm
	}

	actual := DefaultVecOpsConfig()

	assert.Equal(t, expected, actual)
}
