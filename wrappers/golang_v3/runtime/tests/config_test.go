package test

import (
	"fmt"
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"

	"github.com/stretchr/testify/assert"
)

func TestConfigExtensionIntValues(t *testing.T) {
	config := runtime.CreateConfigExtension()
	for i := 0; i < 100; i++ {
		config.SetInt("test", i)
		assert.Equal(t, config.GetInt("test"), i, "result does not match")
	}
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("test%d", i)
		config.SetInt(key, i)
		assert.Equal(t, config.GetInt(key), i, "result does not match")
	}
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("test%d", i)
		config.SetInt(key, i*3)
		assert.Equal(t, config.GetInt(key), i*3, "result does not match")
	}
}

func TestConfigExtensionBoolValues(t *testing.T) {
	config := runtime.CreateConfigExtension()
	for i := 0; i < 100; i++ {
		config.SetBool("test", i%2 == 1)
		assert.Equal(t, config.GetBool("test"), i%2 == 1, "result does not match")
	}
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("test%d", i)
		config.SetBool(key, i%2 == 1)
		assert.Equal(t, config.GetBool(key), i%2 == 1, "result does not match")
	}
}

func TestConfigExtensionSetDifferentValueTypes(t *testing.T) {
	config := runtime.CreateConfigExtension()
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("test%d", i)
		config.SetInt(key, i)
		assert.Equal(t, config.GetInt(key), i, "result does not match")
		config.SetBool(key, i%2 == 1)
		assert.Equal(t, config.GetBool(key), i%2 == 1, "result does not match")
	}
}

func TestConfigExtensionCreateManyConfigs(t *testing.T) {
	var configs [5]runtime.ConfigExtension
	for i := range configs {
		configs[i] = *runtime.CreateConfigExtension()
	}
	for i := 1; i <= 100; i++ {
		key := fmt.Sprintf("test%d", i)
		for j, config := range configs {
			config.SetInt(key, i*(j+1))
		}
	}
	for i := 1; i <= 100; i++ {
		key := fmt.Sprintf("test%d", i)
		for j, config := range configs {
			assert.Equal(t, config.GetInt(key), i*(j+1), "result does not match")
		}
	}
}
