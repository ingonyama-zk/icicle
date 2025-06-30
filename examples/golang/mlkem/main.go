package main

import (
	"crypto/rand"
	"fmt"
	"log"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	mlkem "github.com/ingonyama-zk/icicle/v3/wrappers/golang/pqc/ml-kem"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

const BATCH = 1 << 12

func main() {
	// Allocate buffers on the host
	params := mlkem.Kyber768
	entropyLen := BATCH * mlkem.ENTROPY_BYTES
	messagesLen := BATCH * mlkem.MESSAGE_BYTES
	publicKeysLen := BATCH * params.GetPublicKeyBytes()
	secretKeysLen := BATCH * params.GetSecretKeyBytes()
	ciphertextsLen := BATCH * params.GetCiphertextBytes()
	sharedSecretsLen := BATCH * params.GetSharedSecretBytes()

	entropy := make([]byte, entropyLen)
	messages := make([]byte, messagesLen)
	publicKeys := make([]byte, publicKeysLen)
	secretKeys := make([]byte, secretKeysLen)
	ciphertexts := make([]byte, ciphertextsLen)
	sharedSecretsEnc := make([]byte, sharedSecretsLen)
	sharedSecretsDec := make([]byte, sharedSecretsLen)

	// Fill entropy and messages with randomness
	if _, err := rand.Read(entropy); err != nil {
		log.Fatalf("Failed to generate entropy: %v", err)
	}
	if _, err := rand.Read(messages); err != nil {
		log.Fatalf("Failed to generate messages: %v", err)
	}

	// Configuration – everything stays on host
	config := mlkem.GetDefaultMlKemConfig()
	config.BatchSize = BATCH

	// Initialize device
	device := runtime.CreateDevice("CUDA-PQC", 0)
	// NOTE: If you are only using a single device the entire time
	// 			then this is ok. If you are using multiple devices
	// 			then you should use runtime.RunOnDevice() instead.
	err := runtime.SetDefaultDevice(&device)
	if err != runtime.Success {
		log.Fatalf("Failed to set device: %v", err)
	}

	// Key generation
	err = mlkem.Keygen(params,
		core.HostSliceFromElements(entropy),
		config,
		core.HostSliceFromElements(publicKeys),
		core.HostSliceFromElements(secretKeys),
	)
	if err != runtime.Success {
		log.Fatalf("Keygen failed: %v", err)
	}

	// Encapsulation
	err = mlkem.Encapsulate(params,
		core.HostSliceFromElements(messages),
		core.HostSliceFromElements(publicKeys),
		config,
		core.HostSliceFromElements(ciphertexts),
		core.HostSliceFromElements(sharedSecretsEnc),
	)
	if err != runtime.Success {
		log.Fatalf("Encapsulate failed: %v", err)
	}

	// Decapsulation
	err = mlkem.Decapsulate(params,
		core.HostSliceFromElements(secretKeys),
		core.HostSliceFromElements(ciphertexts),
		config,
		core.HostSliceFromElements(sharedSecretsDec),
	)
	if err != runtime.Success {
		log.Fatalf("Decapsulate failed: %v", err)
	}

	// Verify shared secrets match
	for i := 0; i < len(sharedSecretsEnc); i++ {
		if sharedSecretsEnc[i] != sharedSecretsDec[i] {
			log.Fatalf("Shared secrets do not match at index %d", i)
		}
	}

	fmt.Printf("%d successful KEM operations!\n", BATCH)
}
