package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

	func testCreateStream(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()

	_, err := runtime.CreateStream()
	suite.Equal(runtime.Success, err, "Unable to create stream due to %d", err)
}

func testDestroyStream(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()

	stream, err := runtime.CreateStream()
	suite.Equal(runtime.Success, err, "Unable to create stream due to %d", err)

	err = runtime.DestroyStream(stream)
	suite.Equal(runtime.Success, err, "Unable to destroy stream due to %d", err)
}

func testSyncStream(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()

	stream, err := runtime.CreateStream()
	suite.Equal(runtime.Success, err, "Unable to create stream due to %d", err)

	_, err = runtime.MallocAsync(200000, stream)
	suite.Equal(runtime.Success, err, "Unable to allocate device memory due to %d", err)

	dp, err := runtime.Malloc(20)
	suite.NotNil(dp)
	suite.Equal(runtime.Success, err, "Unable to allocate device memory due to %d", err)

	err = runtime.SynchronizeStream(stream)
	suite.Equal(runtime.Success, err, "Unable to sync stream due to %d", err)

	err = runtime.DestroyStream(stream)
	suite.Equal(runtime.Success, err, "Unable to destroy stream due to %d", err)
}

type StreamTestSuite struct {
	suite.Suite
}

func (s *StreamTestSuite) TestCreateStream() {
	s.Run("TestCreateStream", test_helpers.TestWrapper(&s.Suite, testCreateStream))
	s.Run("TestDestroyStream", test_helpers.TestWrapper(&s.Suite, testDestroyStream))
	s.Run("TestSyncStream", test_helpers.TestWrapper(&s.Suite, testSyncStream))
}

func TestSuiteStream(t *testing.T) {
	suite.Run(t, new(StreamTestSuite))
}
