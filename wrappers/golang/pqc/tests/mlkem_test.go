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

func (s *mlkemTestSuite) runConsistencyDeviceAsync(params *mlkem.KyberParams, batchSize int) {
	// --- Host buffers ---
	publicKeysLen := batchSize * params.PublicKeyBytes
	secretKeysLen := batchSize * params.SecretKeyBytes
	ciphertextsLen := batchSize * params.CiphertextBytes
	sharedSecretsLen := batchSize * params.SharedSecretBytes
	messagesLen := batchSize * mlkem.MESSAGE_BYTES
	entropyLen := batchSize * mlkem.ENTROPY_BYTES

	hostEntropy := make([]byte, entropyLen)
	hostMessages := make([]byte, messagesLen)
	if _, err := rand.Read(hostEntropy); err != nil {
		s.FailNow(fmt.Sprintf("could not fill entropy: %v", err))
	}
	if _, err := rand.Read(hostMessages); err != nil {
		s.FailNow(fmt.Sprintf("could not fill messages: %v", err))
	}

	// Host slots for results
	hostSharedEnc := make([]byte, sharedSecretsLen)
	hostSharedDec := make([]byte, sharedSecretsLen)

	// --- Device setup ---
	device := runtime.CreateDevice("CUDA-PQC", 0)
	runtime.SetDevice(&device)

	cfg := mlkem.GetDefaultMlKemConfig()
	cfg.BatchSize = uint64(batchSize)
	cfg.IsAsync = true
	stream, err := runtime.CreateStream()
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("CreateStream failed: %v", err))
	}
	cfg.StreamHandle = stream

	// Allocate device slices
	var dEntropy, dMessages, dPublicKeys, dSecretKeys, dCiphertexts, dSharedEnc, dSharedDec core.DeviceSlice

	// Copy host -> device
	core.HostSliceFromElements(hostEntropy).CopyToDeviceAsync(&dEntropy, stream, true)
	core.HostSliceFromElements(hostMessages).CopyToDeviceAsync(&dMessages, stream, true)

	dPublicKeys.MallocAsync(1, publicKeysLen, stream)
	dSecretKeys.MallocAsync(1, secretKeysLen, stream)
	dCiphertexts.MallocAsync(1, ciphertextsLen, stream)
	dSharedEnc.MallocAsync(1, sharedSecretsLen, stream)
	dSharedDec.MallocAsync(1, sharedSecretsLen, stream)

	// --- Keygen on device ---
	err = mlkem.Keygen(params,
		dEntropy,
		cfg,
		dPublicKeys,
		dSecretKeys,
	)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("Keygen failed: %v", err))
	}

	// --- Encapsulate on device ---
	err = mlkem.Encapsulate(params,
		dMessages,
		dPublicKeys,
		cfg,
		dCiphertexts,
		dSharedEnc,
	)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("Encapsulate failed: %v", err))
	}

	// --- Decapsulate on device ---
	err = mlkem.Decapsulate(params,
		dSecretKeys,
		dCiphertexts,
		cfg,
		dSharedDec,
	)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("Decapsulate failed: %v", err))
	}

	// Copy results back host <- device
	core.HostSliceFromElements(hostSharedEnc).CopyFromDeviceAsync(&dSharedEnc, stream)
	core.HostSliceFromElements(hostSharedDec).CopyFromDeviceAsync(&dSharedDec, stream)

	// Wait for everything
	runtime.SynchronizeStream(stream)
	runtime.DestroyStream(stream)

	// Compare
	s.Equal(hostSharedEnc, hostSharedDec, "shared secrets should match on device async")
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

func (s *mlkemTestSuite) TestKyberConsistencyDeviceAsync() {
	batch_size := 1 << 13
	s.Run("Kyber512", test_helpers.TestWrapper(&s.Suite, func(_ *suite.Suite) {
		s.runConsistencyDeviceAsync(&mlkem.Kyber512Params, batch_size)
	}))
	s.Run("Kyber768", test_helpers.TestWrapper(&s.Suite, func(_ *suite.Suite) {
		s.runConsistencyDeviceAsync(&mlkem.Kyber768Params, batch_size)
	}))
	s.Run("Kyber1024", test_helpers.TestWrapper(&s.Suite, func(_ *suite.Suite) {
		s.runConsistencyDeviceAsync(&mlkem.Kyber1024Params, batch_size)
	}))
}

func TestSuitemlkem(t *testing.T) {

	suite.Run(t, new(mlkemTestSuite))
}
