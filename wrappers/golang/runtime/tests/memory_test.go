package tests

import (
	"testing"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testMalloc(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()
	mem, err := runtime.Malloc(20)
	suite.Equal(runtime.Success, err, "Unable to allocate device memory due to %d", err)
	runtime.Free(mem)
}

func testMallocAsync(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()
	stream, _ := runtime.CreateStream()
	mem, err := runtime.MallocAsync(20, stream)
	suite.Equal(runtime.Success, err, "Unable to allocate device memory due to %d", err)

	runtime.FreeAsync(mem, stream)
	runtime.SynchronizeStream(stream)
	runtime.DestroyStream(stream)
}

func testFree(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()
	mem, err := runtime.Malloc(20)
	suite.Equal(runtime.Success, err, "Unable to allocate device memory due to %d", err)

	err = runtime.Free(mem)
	suite.Equal(runtime.Success, err, "Unable to free device memory due to %v", err)
}

func testCopyFromToHost(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()
	someInts := make([]int32, 1)
	someInts[0] = 34
	numBytes := uint(4)
	deviceMem, _ := runtime.Malloc(numBytes)
	deviceMem, err := runtime.CopyToDevice(deviceMem, unsafe.Pointer(&someInts[0]), numBytes)
	suite.Equal(runtime.Success, err, "Couldn't copy to device due to %v", err)

	someInts2 := make([]int32, 1)
	_, err = runtime.CopyFromDevice(unsafe.Pointer(&someInts2[0]), deviceMem, numBytes)
	suite.Equal(runtime.Success, err, "Couldn't copy to device due to %v", err)
	suite.Equal(someInts, someInts2, "Elements of host slices do not match. Copying from/to host failed")
	runtime.Free(deviceMem)
}

type MemoryTestSuite struct {
	suite.Suite
}

func (s *MemoryTestSuite) TestMemory() {
	s.Run("TestMalloc", test_helpers.TestWrapper(&s.Suite, testMalloc))
	s.Run("TestMallocAsync", test_helpers.TestWrapper(&s.Suite, testMallocAsync))
	s.Run("TestFree", test_helpers.TestWrapper(&s.Suite, testFree))
	s.Run("TestCopyFromToHost", test_helpers.TestWrapper(&s.Suite, testCopyFromToHost))
}

func TestSuiteMemory(t *testing.T) {
	suite.Run(t, new(MemoryTestSuite))
}
