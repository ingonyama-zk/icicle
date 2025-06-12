package ml_kem

// #cgo CFLAGS: -I./include/
// #include "mlkem.h"
import "C"

import "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

const (
	ENTROPY_BYTES = 64
	MESSAGE_BYTES = 32
)

type KyberParams struct {
	PublicKeyBytes    int
	SecretKeyBytes    int
	CiphertextBytes   int
	SharedSecretBytes int
	K                 uint8
	Eta1              uint8
	Eta2              uint8
	Du                uint8
	Dv                uint8
	KeygenFFI         func(cEntropy *C.uint8_t, cConfig *C.MlKemConfig, cPublicKeys *C.uint8_t, cSecretKeys *C.uint8_t) runtime.EIcicleError
	EncapsulateFFI    func(cMessage *C.uint8_t, cPublicKeys *C.uint8_t, cConfig *C.MlKemConfig, cCiphertexts *C.uint8_t, cSharedSecrets *C.uint8_t) runtime.EIcicleError
	DecapsulateFFI    func(cSecretKeys *C.uint8_t, cCiphertexts *C.uint8_t, cConfig *C.MlKemConfig, cSharedSecrets *C.uint8_t) runtime.EIcicleError
}

var Kyber512Params = KyberParams{
	PublicKeyBytes:    800,
	SecretKeyBytes:    1632,
	CiphertextBytes:   768,
	SharedSecretBytes: 32,
	K:                 2,
	Eta1:              3,
	Eta2:              2,
	Du:                10,
	Dv:                4,

	KeygenFFI: func(cEntropy *C.uint8_t, cConfig *C.MlKemConfig, cPublicKeys *C.uint8_t, cSecretKeys *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_keygen512(cEntropy, cConfig, cPublicKeys, cSecretKeys))
	},
	EncapsulateFFI: func(cMessage *C.uint8_t, cPublicKeys *C.uint8_t, cConfig *C.MlKemConfig, cCiphertexts *C.uint8_t, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_encapsulate512(cMessage, cPublicKeys, cConfig, cCiphertexts, cSharedSecrets))
	},
	DecapsulateFFI: func(cSecretKeys *C.uint8_t, cCiphertexts *C.uint8_t, cConfig *C.MlKemConfig, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_decapsulate512(cSecretKeys, cCiphertexts, cConfig, cSharedSecrets))
	},
}

var Kyber768Params = KyberParams{
	PublicKeyBytes:    1184,
	SecretKeyBytes:    2400,
	CiphertextBytes:   1088,
	SharedSecretBytes: 32,
	K:                 3,
	Eta1:              2,
	Eta2:              2,
	Du:                10,
	Dv:                4,

	KeygenFFI: func(cEntropy *C.uint8_t, cConfig *C.MlKemConfig, cPublicKeys *C.uint8_t, cSecretKeys *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_keygen768(cEntropy, cConfig, cPublicKeys, cSecretKeys))
	},
	EncapsulateFFI: func(cMessage *C.uint8_t, cPublicKeys *C.uint8_t, cConfig *C.MlKemConfig, cCiphertexts *C.uint8_t, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_encapsulate768(cMessage, cPublicKeys, cConfig, cCiphertexts, cSharedSecrets))
	},
	DecapsulateFFI: func(cSecretKeys *C.uint8_t, cCiphertexts *C.uint8_t, cConfig *C.MlKemConfig, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_decapsulate768(cSecretKeys, cCiphertexts, cConfig, cSharedSecrets))
	},
}

var Kyber1024Params = KyberParams{
	PublicKeyBytes:    1568,
	SecretKeyBytes:    3168,
	CiphertextBytes:   1568,
	SharedSecretBytes: 32,
	K:                 4,
	Eta1:              2,
	Eta2:              2,
	Du:                11,
	Dv:                5,

	KeygenFFI: func(cEntropy *C.uint8_t, cConfig *C.MlKemConfig, cPublicKeys *C.uint8_t, cSecretKeys *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_keygen1024(cEntropy, cConfig, cPublicKeys, cSecretKeys))
	},
	EncapsulateFFI: func(cMessage *C.uint8_t, cPublicKeys *C.uint8_t, cConfig *C.MlKemConfig, cCiphertexts *C.uint8_t, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_encapsulate1024(cMessage, cPublicKeys, cConfig, cCiphertexts, cSharedSecrets))
	},
	DecapsulateFFI: func(cSecretKeys *C.uint8_t, cCiphertexts *C.uint8_t, cConfig *C.MlKemConfig, cSharedSecrets *C.uint8_t) runtime.EIcicleError {
		return runtime.EIcicleError(C.icicle_ml_kem_decapsulate1024(cSecretKeys, cCiphertexts, cConfig, cSharedSecrets))
	},
}
