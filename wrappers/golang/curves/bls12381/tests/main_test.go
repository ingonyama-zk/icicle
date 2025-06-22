//go:build !ntt

package tests

import (
	"os"
	"sync"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
	bls12_381 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

const (
	largestTestSize = 20
)

func TestMain(m *testing.M) {
	test_helpers.LOAD_AND_INIT_MAIN_DEVICE()

	os.Exit(m.Run())
}
