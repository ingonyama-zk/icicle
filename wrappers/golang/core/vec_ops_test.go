package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVecOpsDefaultConfig(t *testing.T) {
	actual := DefaultVecOpsConfig()
	expected := VecOpsConfig{
		actual.StreamHandle, // Ctx
		false,               // isAOnDevice
		false,               // isBOnDevice
		false,               // isResultOnDevice
		false,               // IsAsync
		1,                   // BatchSize
		false,               // ColumnsBatch
		actual.Ext,          // Ext
	}

	assert.Equal(t, expected, actual)
}
