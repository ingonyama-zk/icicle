package mlkem

// #cgo CFLAGS: -I./include/
// #include "mlkem.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func Keygen(params *KyberParams, entropy core.HostOrDeviceSlice, config MlKemConfig, publicKeys core.HostOrDeviceSlice, secretKeys core.HostOrDeviceSlice) runtime.EIcicleError {
	err := KeygenCheck(entropy, publicKeys, secretKeys, &config, params)
	if err != runtime.Success {
		return err
	}

	cEntropy := (*C.uint8_t)(entropy.AsUnsafePointer())
	cConfig := (*C.MlKemConfig)(unsafe.Pointer(&config))
	cPublicKeys := (*C.uint8_t)(publicKeys.AsUnsafePointer())
	cSecretKeys := (*C.uint8_t)(secretKeys.AsUnsafePointer())

	__ret := params.KeygenFFI(cEntropy, cConfig, cPublicKeys, cSecretKeys)
	return runtime.EIcicleError(__ret)
}

func Encapsulate(params *KyberParams, message core.HostOrDeviceSlice, publicKeys core.HostOrDeviceSlice, config MlKemConfig, ciphertexts core.HostOrDeviceSlice, sharedSecrets core.HostOrDeviceSlice) runtime.EIcicleError {
	err := EncapsCheck(message, publicKeys, ciphertexts, sharedSecrets, &config, params)
	if err != runtime.Success {
		return err
	}

	cMessage := (*C.uint8_t)(message.AsUnsafePointer())
	cPublicKeys := (*C.uint8_t)(publicKeys.AsUnsafePointer())
	cConfig := (*C.MlKemConfig)(unsafe.Pointer(&config))
	cCiphertexts := (*C.uint8_t)(ciphertexts.AsUnsafePointer())
	cSharedSecrets := (*C.uint8_t)(sharedSecrets.AsUnsafePointer())

	__ret := params.EncapsulateFFI(cMessage, cPublicKeys, cConfig, cCiphertexts, cSharedSecrets)
	return runtime.EIcicleError(__ret)
}

func Decapsulate(params *KyberParams, secretKeys core.HostOrDeviceSlice, ciphertexts core.HostOrDeviceSlice, config MlKemConfig, sharedSecrets core.HostOrDeviceSlice) runtime.EIcicleError {
	err := DecapsCheck(secretKeys, ciphertexts, sharedSecrets, &config, params)
	if err != runtime.Success {
		return err
	}

	cSecretKeys := (*C.uint8_t)(secretKeys.AsUnsafePointer())
	cCiphertexts := (*C.uint8_t)(ciphertexts.AsUnsafePointer())
	cConfig := (*C.MlKemConfig)(unsafe.Pointer(&config))
	cSharedSecrets := (*C.uint8_t)(sharedSecrets.AsUnsafePointer())

	__ret := params.DecapsulateFFI(cSecretKeys, cCiphertexts, cConfig, cSharedSecrets)
	return runtime.EIcicleError(__ret)
}
