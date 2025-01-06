package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func TestMain(m *testing.M) {
	runtime.LoadBackendFromEnvOrDefault()
	m.Run()
}
