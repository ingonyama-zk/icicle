package tests

import (
	"os"
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
)

const (
	largestTestSize = 20
)

func TestMain(m *testing.M) {
	runtime.LoadBackendFromEnv()
	device := runtime.CreateDevice("CUDA", 0)
	runtime.SetDevice(&device)

	// execute tests
	os.Exit(m.Run())

}
