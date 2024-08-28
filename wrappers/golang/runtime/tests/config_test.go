package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"

	"github.com/stretchr/testify/assert"
)

func TestConfigExtensionIntValues(t *testing.T) {
	config := config_extension.CreateConfigExtension()
	config.SetInt("test", 5)
	assert.Equal(t, config.GetInt("test"), 5, "result does not match")
	config.SetInt("test", 99)
	assert.Equal(t, config.GetInt("test"), 99, "result does not match")
}

func TestConfigExtensionBoolValues(t *testing.T) {
	config := config_extension.CreateConfigExtension()
	config.SetBool("test", true)
	assert.Equal(t, config.GetBool("test"), true, "result does not match")
	config.SetBool("test", false)
	assert.Equal(t, config.GetBool("test"), false, "result does not match")
}

func TestConfigExtensionSetDifferentValueTypes(t *testing.T) {
	config := config_extension.CreateConfigExtension()

	config.SetInt("test", 5)
	assert.Equal(t, config.GetInt("test"), 5, "result does not match")
	config.SetBool("test", true)
	assert.Equal(t, config.GetBool("test"), true, "result does not match")
}
