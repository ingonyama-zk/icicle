package mlkem

// #cgo CFLAGS: -I./include/
// #include "mlkem.h"
import "C"

import "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

const (
	ENTROPY_BYTES = 64
	MESSAGE_BYTES = 32
)

type KyberMode int

const (
	Kyber512 KyberMode = iota
	Kyber768
	Kyber1024
)

type kyberParams struct {
	publicKeyBytes    int
	secretKeyBytes    int
	ciphertextBytes   int
	sharedSecretBytes int

	keygenFFI      func(*C.uint8_t, *C.MlKemConfig, *C.uint8_t, *C.uint8_t) runtime.EIcicleError
	encapsulateFFI func(*C.uint8_t, *C.uint8_t, *C.MlKemConfig, *C.uint8_t, *C.uint8_t) runtime.EIcicleError
	decapsulateFFI func(*C.uint8_t, *C.uint8_t, *C.MlKemConfig, *C.uint8_t) runtime.EIcicleError
}

func (mode KyberMode) getParams() kyberParams {
	switch mode {
	case Kyber512:
		return kyberParams{
			publicKeyBytes:    800,
			secretKeyBytes:    1632,
			ciphertextBytes:   768,
			sharedSecretBytes: 32,

			keygenFFI: func(cEntropy *C.uint8_t, cConfig *C.MlKemConfig, cPublicKeys *C.uint8_t, cSecretKeys *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_keygen512(cEntropy, cConfig, cPublicKeys, cSecretKeys))
			},
			encapsulateFFI: func(cMessage *C.uint8_t, cPublicKeys *C.uint8_t, cConfig *C.MlKemConfig, cCiphertexts *C.uint8_t, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_encapsulate512(cMessage, cPublicKeys, cConfig, cCiphertexts, cSharedSecrets))
			},
			decapsulateFFI: func(cSecretKeys *C.uint8_t, cCiphertexts *C.uint8_t, cConfig *C.MlKemConfig, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_decapsulate512(cSecretKeys, cCiphertexts, cConfig, cSharedSecrets))
			},
		}
	case Kyber768:
		return kyberParams{
			publicKeyBytes:    1184,
			secretKeyBytes:    2400,
			ciphertextBytes:   1088,
			sharedSecretBytes: 32,

			keygenFFI: func(cEntropy *C.uint8_t, cConfig *C.MlKemConfig, cPublicKeys *C.uint8_t, cSecretKeys *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_keygen768(cEntropy, cConfig, cPublicKeys, cSecretKeys))
			},
			encapsulateFFI: func(cMessage *C.uint8_t, cPublicKeys *C.uint8_t, cConfig *C.MlKemConfig, cCiphertexts *C.uint8_t, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_encapsulate768(cMessage, cPublicKeys, cConfig, cCiphertexts, cSharedSecrets))
			},
			decapsulateFFI: func(cSecretKeys *C.uint8_t, cCiphertexts *C.uint8_t, cConfig *C.MlKemConfig, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_decapsulate768(cSecretKeys, cCiphertexts, cConfig, cSharedSecrets))
			},
		}
	case Kyber1024:
		return kyberParams{
			publicKeyBytes:    1568,
			secretKeyBytes:    3168,
			ciphertextBytes:   1568,
			sharedSecretBytes: 32,

			keygenFFI: func(cEntropy *C.uint8_t, cConfig *C.MlKemConfig, cPublicKeys *C.uint8_t, cSecretKeys *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_keygen1024(cEntropy, cConfig, cPublicKeys, cSecretKeys))
			},
			encapsulateFFI: func(cMessage *C.uint8_t, cPublicKeys *C.uint8_t, cConfig *C.MlKemConfig, cCiphertexts *C.uint8_t, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_encapsulate1024(cMessage, cPublicKeys, cConfig, cCiphertexts, cSharedSecrets))
			},
			decapsulateFFI: func(cSecretKeys *C.uint8_t, cCiphertexts *C.uint8_t, cConfig *C.MlKemConfig, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
				return runtime.EIcicleError(C.icicle_ml_kem_decapsulate1024(cSecretKeys, cCiphertexts, cConfig, cSharedSecrets))
			},
		}
	default:
		panic("invalid KyberMode")
	}
}

func (mode KyberMode) GetPublicKeyBytes() int {
	return mode.getParams().publicKeyBytes
}

func (mode KyberMode) GetSecretKeyBytes() int {
	return mode.getParams().secretKeyBytes
}

func (mode KyberMode) GetCiphertextBytes() int {
	return mode.getParams().ciphertextBytes
}

func (mode KyberMode) GetSharedSecretBytes() int {
	return mode.getParams().sharedSecretBytes
}

func (mode KyberMode) getKeygenFFI() func(*C.uint8_t, *C.MlKemConfig, *C.uint8_t, *C.uint8_t) runtime.EIcicleError {
	return mode.getParams().keygenFFI
}

func (mode KyberMode) getEncapsulateFFI() func(*C.uint8_t, *C.uint8_t, *C.MlKemConfig, *C.uint8_t, *C.uint8_t) runtime.EIcicleError {
	return mode.getParams().encapsulateFFI
}

func (mode KyberMode) getDecapsulateFFI() func(*C.uint8_t, *C.uint8_t, *C.MlKemConfig, *C.uint8_t) runtime.EIcicleError {
	return mode.getParams().decapsulateFFI
}
