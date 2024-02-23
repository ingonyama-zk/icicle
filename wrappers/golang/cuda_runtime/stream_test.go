package cuda_runtime

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCreateStream(t *testing.T) {
	_, err := CreateStream()
	assert.Equal(t, CudaSuccess, err, "Unable to create stream due to %d", err)
}

func TestCreateStreamWithFlags(t *testing.T) {
	_, err := CreateStreamWithFlags(CudaStreamDefault)
	assert.Equal(t, CudaSuccess, err, "Unable to create stream due to %d", err)

	_, errNonBlocking := CreateStreamWithFlags(CudaStreamNonBlocking)
	assert.Equal(t, CudaSuccess, errNonBlocking, "Unable to create stream due to %d", errNonBlocking)
}

func TestDestroyStream(t *testing.T) {
	stream, err := CreateStream()
	assert.Equal(t, CudaSuccess, err, "Unable to create stream due to %d", err)

	err = DestroyStream(&stream)
	assert.Equal(t, CudaSuccess, err, "Unable to destroy stream due to %d", err)
}

func TestSyncStream(t *testing.T) {
	stream, err := CreateStream()
	assert.Equal(t, CudaSuccess, err, "Unable to create stream due to %d", err)

	_, err = MallocAsync(200000, stream)
	assert.Equal(t, CudaSuccess, err, "Unable to allocate device memory due to %d", err)

	dp, err := Malloc(20)
	assert.NotNil(t, dp)
	assert.Equal(t, CudaSuccess, err, "Unable to allocate device memory due to %d", err)

	err = SynchronizeStream(&stream)
	assert.Equal(t, CudaSuccess, err, "Unable to sync stream due to %d", err)

	err = DestroyStream(&stream)
	assert.Equal(t, CudaSuccess, err, "Unable to destroy stream due to %d", err)
}
