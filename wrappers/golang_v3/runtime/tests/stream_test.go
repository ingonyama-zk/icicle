package test

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
	"github.com/stretchr/testify/assert"
)

func TestCreateStream(t *testing.T) {
	err := runtime.LoadBackendFromEnv()
	assert.Equal(t, runtime.Success, err)
	dev := runtime.CreateDevice("CUDA", 0)
	assert.True(t, runtime.IsDeviceAvailable(&dev))
	err = runtime.SetDevice(&dev)
	assert.Equal(t, runtime.Success, err)
	_, err = runtime.CreateStream()
	assert.Equal(t, runtime.Success, err, "Unable to create stream due to %d", err)
}

func TestDestroyStream(t *testing.T) {
	err := runtime.LoadBackend("/home/administrator/users/Timur/Projects/icicle/icicle_v3/build/backend", true)
	assert.Equal(t, runtime.Success, err)
	dev := runtime.CreateDevice("CUDA", 0)
	assert.True(t, runtime.IsDeviceAvailable(&dev))
	stream, err := runtime.CreateStream()
	assert.Equal(t, runtime.Success, err, "Unable to create stream due to %d", err)

	err = runtime.DestroyStream(stream)
	assert.Equal(t, runtime.Success, err, "Unable to destroy stream due to %d", err)
}

func TestSyncStream(t *testing.T) {
	err := runtime.LoadBackendFromEnv()
	assert.Equal(t, runtime.Success, err)
	dev := runtime.CreateDevice("CUDA", 0)
	assert.True(t, runtime.IsDeviceAvailable(&dev))
	runtime.SetDevice(&dev)

	stream, err := runtime.CreateStream()
	assert.Equal(t, runtime.Success, err, "Unable to create stream due to %d", err)

	_, err = runtime.MallocAsync(200000, stream)
	assert.Equal(t, runtime.Success, err, "Unable to allocate device memory due to %d", err)

	dp, err := runtime.Malloc(20)
	assert.NotNil(t, dp)
	assert.Equal(t, runtime.Success, err, "Unable to allocate device memory due to %d", err)

	err = runtime.SynchronizeStream(stream)
	assert.Equal(t, runtime.Success, err, "Unable to sync stream due to %d", err)

	err = runtime.DestroyStream(stream)
	assert.Equal(t, runtime.Success, err, "Unable to destroy stream due to %d", err)
}
