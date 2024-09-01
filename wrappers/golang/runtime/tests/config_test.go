package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"

	"github.com/stretchr/testify/assert"
)

func TestConfigExtensionIntValues(t *testing.T) {
	config := config_extension.Create()
	config.SetInt("test", 5)
	assert.Equal(t, 5, config.GetInt("test"), "result does not match")
	config.SetInt("test", 99)
	assert.Equal(t, 99, config.GetInt("test"), "result does not match")
}

func TestConfigExtensionBoolValues(t *testing.T) {
	config := config_extension.Create()
	config.SetBool("test", true)
	assert.True(t, config.GetBool("test"), "result does not match")
	config.SetBool("test", false)
	assert.False(t, config.GetBool("test"), "result does not match")
}

func TestConfigExtensionSetDifferentValueTypes(t *testing.T) {
	config := config_extension.Create()

	config.SetInt("test", 5)
	assert.Equal(t, 5, config.GetInt("test"), "result does not match")
	config.SetBool("test", true)
	assert.True(t, config.GetBool("test"), "result does not match")
}
