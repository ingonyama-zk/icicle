package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVecOpsDefaultConfig(t *testing.T) {
	actual := DefaultVecOpsConfig()
	expected := VecOpsConfig{
		actual.StreamHandle, // Ctx
		true,                // isAOnDevice
		true,                // isBOnDevice
		true,                // isResultOnDevice
		true,                // IsAsync
		actual.Ext,          // Ext
	}

	assert.Equal(t, expected, actual)
}
