package test

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"

	"github.com/stretchr/testify/assert"
)

func TestGetDeviceType(t *testing.T) {
	config := runtime.CreateDevice("test", 0)
	assert.Equal(t, config.GetDeviceType(), "test")
	configLargeName := runtime.CreateDevice("testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttest", 1)
	assert.Equal(t, configLargeName.GetDeviceType(), "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttes")
}
