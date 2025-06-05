package tests

import (
	"crypto/rand"
	"fmt"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testKeccakBatch(s *suite.Suite) {
	single_hash_input_size := 30
	batch := 3
	const outputBytes = 64 // 64 bytes is output size of Keccak512

	input := make([]byte, single_hash_input_size*batch)
	_, err := rand.Read(input)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	keccakHasher, error := hash.NewKeccak512Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes*batch)
	keccakHasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	test_helpers.ActivateMainDevice()
	keccakHasher, error = hash.NewKeccak512Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputMain := make([]byte, outputBytes*batch)
	keccakHasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputMain),
		core.GetDefaultHashConfig(),
	)

	outputEmpty := make([]byte, outputBytes*batch)
	s.Equal(outputRef, outputMain)
	s.NotEqual(outputEmpty, outputMain)
}

func testBlake2s(s *suite.Suite) {
	singleHashInputSize := 567
	batch := 11
	const outputBytes = 32 // 32 bytes is output size of Blake2s

	input := make([]byte, singleHashInputSize*batch)
	_, err := rand.Read(input)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	Blake2sHasher, error := hash.NewBlake2sHasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes*batch)
	Blake2sHasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	test_helpers.ActivateMainDevice()
	Blake2sHasher, error = hash.NewBlake2sHasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputMain := make([]byte, outputBytes*batch)
	Blake2sHasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputMain),
		core.GetDefaultHashConfig(),
	)

	outputEmpty := make([]byte, outputBytes*batch)
	s.Equal(outputRef, outputMain)
	s.NotEqual(outputEmpty, outputMain)
}

func testBlake3_cpu_gpu(s *suite.Suite) {
	singleHashInputSize := 567
	batch := 11
	const outputBytes = 32 // 32 bytes is output size of Blake3

	input := make([]byte, singleHashInputSize*batch)
	_, err := rand.Read(input)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	Blake3Hasher, error := hash.NewBlake3Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes*batch)
	Blake3Hasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	test_helpers.ActivateMainDevice()
	Blake3Hasher, error = hash.NewBlake3Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputMain := make([]byte, outputBytes*batch)
	Blake3Hasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputMain),
		core.GetDefaultHashConfig(),
	)

	outputEmpty := make([]byte, outputBytes*batch)
	s.Equal(outputRef, outputMain)
	s.NotEqual(outputEmpty, outputMain)
}

func testBlake3(s *suite.Suite) {
	const outputBytes = 32 // 32 bytes is output size of Blake3

	// Known input string and expected hash
	inputString := "Hello world I am blake3. This is a semi-long Go test with a lot of characters. 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	expectedHash := "a2b794acb5a604bbd2c4c0380e935697e0b934ea6f194b9f5246fbb212ebe549"

	input := []byte(inputString)

	Blake3Hasher, error := hash.NewBlake3Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes)
	Blake3Hasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	outputRefHex := fmt.Sprintf("%x", outputRef)

	s.Equal(expectedHash, outputRefHex, "Hash mismatch: got %s, expected %s", outputRefHex, expectedHash)
}

func testSha3(s *suite.Suite) {
	singleHashInputSize := 1153
	batch := 1
	const outputBytes = 32 // 32 bytes is output size of Sha3 256

	input := make([]byte, singleHashInputSize*batch)
	_, err := rand.Read(input)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	Sha3Hasher, error := hash.NewSha3256Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes*batch)
	Sha3Hasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	test_helpers.ActivateMainDevice()
	Sha3Hasher, error = hash.NewSha3256Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputMain := make([]byte, outputBytes*batch)
	Sha3Hasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputMain),
		core.GetDefaultHashConfig(),
	)

	outputEmpty := make([]byte, outputBytes*batch)
	s.Equal(outputRef, outputMain)
	s.NotEqual(outputEmpty, outputMain)
}

type HashTestSuite struct {
	suite.Suite
}

func (s *HashTestSuite) TestHash() {
	s.Run("TestKeccakBatch", test_helpers.TestWrapper(&s.Suite, testKeccakBatch))
	s.Run("TestBlake2s", test_helpers.TestWrapper(&s.Suite, testBlake2s))
	s.Run("TestBlake3_CPU_GPU", test_helpers.TestWrapper(&s.Suite, testBlake3_cpu_gpu))
	s.Run("TestBlake3", test_helpers.TestWrapper(&s.Suite, testBlake3))
	s.Run("TestSha3", test_helpers.TestWrapper(&s.Suite, testSha3))
}

func TestSuiteHash(t *testing.T) {
	suite.Run(t, new(HashTestSuite))
}
