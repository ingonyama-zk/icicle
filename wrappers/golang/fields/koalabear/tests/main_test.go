//go:build !ntt

package tests

import (
	"os"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
)

const (
	largestTestSize = 20
)

func TestMain(m *testing.M) {
	test_helpers.LOAD_AND_INIT_MAIN_DEVICE()

	os.Exit(m.Run())
}
