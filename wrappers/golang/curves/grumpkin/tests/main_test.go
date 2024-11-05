package tests

import (
	"os"
	"sync"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

const (
	largestTestSize = 20
)

func testWrapper(suite *suite.Suite, fn func(*suite.Suite)) func() {
	return func() {
		wg := sync.WaitGroup{}
		wg.Add(1)
		runtime.RunOnDevice(&test_helpers.REFERENCE_DEVICE, func(args ...any) {
			defer wg.Done()
			fn(suite)
		})
		wg.Wait()
	}
}

func TestMain(m *testing.M) {
	runtime.LoadBackendFromEnvOrDefault()

	exitCode := m.Run()

	os.Exit(exitCode)
}
