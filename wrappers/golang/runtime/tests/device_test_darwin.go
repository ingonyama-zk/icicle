package tests

/*
#include <pthread.h>
#include <stdio.h> // For printing pthread_t if it's an integer type

// On some systems, pthread_t might be a struct or opaque type.
// To print it as an integer, you might need to cast it.
// This is a common way to get a numerical representation for logging/debugging.
unsigned long get_pthread_id() {
    return (unsigned long)pthread_self();
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	icicle_runtime "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/stretchr/testify/suite"
)

func testIsDeviceAvailable(suite *suite.Suite) {
	test_helpers.ActivateMainDevice()
	res, err := icicle_runtime.GetDeviceCount()

	suite.Equal(icicle_runtime.Success, err)
	suite.Equal(1, res) // METAL only has one device

	suite.True(icicle_runtime.IsDeviceAvailable(&test_helpers.MAIN_DEVICE))
	suite.True(icicle_runtime.IsDeviceAvailable(&test_helpers.REFERENCE_DEVICE))
	devInvalid := icicle_runtime.CreateDevice("invalid", 0)
	suite.False(icicle_runtime.IsDeviceAvailable(&devInvalid))
}

func testSetDefaultDevice(suite *suite.Suite) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	
	defaultDevice := &test_helpers.MAIN_DEVICE
	icicle_runtime.SetDefaultDevice(defaultDevice)

	outerThreadID := C.get_pthread_id()
	done := make(chan struct{}, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		// Ensure we are operating on an OS thread other than the original one
		tidInner := C.get_pthread_id()
		for tidInner == outerThreadID {
			fmt.Println("Locked thread is the same as original, getting new locked thread")
			runtime.UnlockOSThread()
			runtime.LockOSThread()
			tidInner = C.get_pthread_id()
		}

		activeDevice, err := icicle_runtime.GetActiveDevice()
		suite.Equal(icicle_runtime.Success, err)
		suite.Equal(defaultDevice, *activeDevice)

		close(done)
	}()

	<-done

	icicle_runtime.SetDefaultDevice(&test_helpers.REFERENCE_DEVICE)
}
