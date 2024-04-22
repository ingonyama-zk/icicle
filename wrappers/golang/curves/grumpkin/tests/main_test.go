package tests

import (
	"os"
	"testing"
)

const (
	largestTestSize = 20
)

func TestMain(m *testing.M) {

	// execute tests
	os.Exit(m.Run())

}
