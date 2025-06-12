package tests

import (
	"crypto/rand"
	"fmt"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	mlkem "github.com/ingonyama-zk/icicle/v3/wrappers/golang/pqc/ml-kem"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

type mlkemTestSuite struct {
	suite.Suite
}

func (s *mlkemTestSuite) runConsistencyHost(params *mlkem.KyberParams, batchSize int) {
	// Allocate buffers
	publicKeysLen := batchSize * params.PublicKeyBytes
	secretKeysLen := batchSize * params.SecretKeyBytes
	ciphertextsLen := batchSize * params.CiphertextBytes
	sharedSecretsLen := batchSize * params.SharedSecretBytes
	messagesLen := batchSize * mlkem.MESSAGE_BYTES
	entropyLen := batchSize * mlkem.ENTROPY_BYTES

	publicKeys := make([]byte, publicKeysLen)
	secretKeys := make([]byte, secretKeysLen)
	messages := make([]byte, messagesLen)
	ciphertexts := make([]byte, ciphertextsLen)
	sharedEnc := make([]byte, sharedSecretsLen)
	sharedDec := make([]byte, sharedSecretsLen)
	entropy := make([]byte, entropyLen)

	// Fill entropy and messages with randomness
	if _, err := rand.Read(entropy); err != nil {
		s.FailNow(fmt.Sprintf("could not fill entropy: %v", err))
	}
	if _, err := rand.Read(messages); err != nil {
		s.FailNow(fmt.Sprintf("could not fill messages: %v", err))
	}

	// Default config + batch size
	config := mlkem.GetDefaultMlKemConfig()
	config.BatchSize = uint64(batchSize)

	device := runtime.CreateDevice("CUDA-PQC", 0)
	runtime.SetDevice(&device)

	// Keygen
	err := mlkem.Keygen(params,
		core.HostSliceFromElements(entropy),
		config,
		core.HostSliceFromElements(publicKeys),
		core.HostSliceFromElements(secretKeys),
	)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("Keygen failed: %v", err))
	}

	// Encapsulate
	err = mlkem.Encapsulate(params,
		core.HostSliceFromElements(messages),
		core.HostSliceFromElements(publicKeys),
		config,
		core.HostSliceFromElements(ciphertexts),
		core.HostSliceFromElements(sharedEnc),
	)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("Encapsulate failed: %v", err))
	}

	// Decapsulate
	err = mlkem.Decapsulate(params,
		core.HostSliceFromElements(secretKeys),
		core.HostSliceFromElements(ciphertexts),
		config,
		core.HostSliceFromElements(sharedDec),
	)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("Decapsulate failed: %v", err))
	}

	// Check equality
	s.Equal(sharedEnc, sharedDec, "shared secrets should match")
}

func (s *mlkemTestSuite) TestKyberConsistencyHost() {
	batch_size := 1 << 13
	s.Run("Kyber512", test_helpers.TestWrapper(&s.Suite, func(_ *suite.Suite) {
		s.runConsistencyHost(&mlkem.Kyber512Params, batch_size)
	}))
	s.Run("Kyber768", test_helpers.TestWrapper(&s.Suite, func(_ *suite.Suite) {
		s.runConsistencyHost(&mlkem.Kyber768Params, batch_size)
	}))
	s.Run("Kyber1024", test_helpers.TestWrapper(&s.Suite, func(_ *suite.Suite) {
		s.runConsistencyHost(&mlkem.Kyber1024Params, batch_size)
	}))
}

func TestSuitemlkem(t *testing.T) {
	suite.Run(t, new(mlkemTestSuite))
}
