package ml_kem

// #cgo CFLAGS: -I./include/
// #include "mlkem.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func Keygen(params *KyberParams, entropy core.HostOrDeviceSlice, config MlKemConfig, publicKeys core.HostOrDeviceSlice, secretKeys core.HostOrDeviceSlice) runtime.EIcicleError {
	if uint64(entropy.Len()) != config.BatchSize*ENTROPY_BYTES {
		return runtime.InvalidArgument
	}
	if uint64(publicKeys.Len()) != config.BatchSize*uint64(params.PublicKeyBytes) {
		return runtime.InvalidArgument
	}
	if uint64(secretKeys.Len()) != config.BatchSize*uint64(params.SecretKeyBytes) {
		return runtime.InvalidArgument
	}
	config.EntropyOnDevice = entropy.IsOnDevice()
	config.PublicKeysOnDevice = publicKeys.IsOnDevice()
	config.SecretKeysOnDevice = secretKeys.IsOnDevice()
	if config.EntropyOnDevice {
		if dev, ok := entropy.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}

	if config.PublicKeysOnDevice {
		if dev, ok := publicKeys.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}

	if config.SecretKeysOnDevice {
		if dev, ok := secretKeys.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}

	cEntropy := (*C.uint8_t)(entropy.AsUnsafePointer())
	cConfig := (*C.MlKemConfig)(unsafe.Pointer(&config))
	cPublicKeys := (*C.uint8_t)(publicKeys.AsUnsafePointer())
	cSecretKeys := (*C.uint8_t)(secretKeys.AsUnsafePointer())

	__ret := params.KeygenFFI(cEntropy, cConfig, cPublicKeys, cSecretKeys)

	err := runtime.EIcicleError(__ret)
	return err
}

func Encapsulate(params *KyberParams, message core.HostOrDeviceSlice, publicKeys core.HostOrDeviceSlice, config MlKemConfig, ciphertexts core.HostOrDeviceSlice, sharedSecrets core.HostOrDeviceSlice) runtime.EIcicleError {
	if uint64(message.Len()) != config.BatchSize*uint64(MESSAGE_BYTES) {
		return runtime.InvalidArgument
	}
	if uint64(publicKeys.Len()) != config.BatchSize*uint64(params.PublicKeyBytes) {
		return runtime.InvalidArgument
	}
	if uint64(ciphertexts.Len()) != config.BatchSize*uint64(params.CiphertextBytes) {
		return runtime.InvalidArgument
	}
	if uint64(sharedSecrets.Len()) != config.BatchSize*uint64(params.SharedSecretBytes) {
		return runtime.InvalidArgument
	}

	config.MessagesOnDevice = message.IsOnDevice()
	config.PublicKeysOnDevice = publicKeys.IsOnDevice()
	config.CiphertextsOnDevice = ciphertexts.IsOnDevice()
	config.SharedSecretsOnDevice = sharedSecrets.IsOnDevice()

	if config.MessagesOnDevice {
		if dev, ok := message.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}
	if config.PublicKeysOnDevice {
		if dev, ok := publicKeys.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}
	if config.CiphertextsOnDevice {
		if dev, ok := ciphertexts.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}
	if config.SharedSecretsOnDevice {
		if dev, ok := sharedSecrets.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
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
	if uint64(secretKeys.Len()) != config.BatchSize*uint64(params.SecretKeyBytes) {
		return runtime.InvalidArgument
	}
	if uint64(ciphertexts.Len()) != config.BatchSize*uint64(params.CiphertextBytes) {
		return runtime.InvalidArgument
	}
	if uint64(sharedSecrets.Len()) != config.BatchSize*uint64(params.SharedSecretBytes) {
		return runtime.InvalidArgument
	}

	config.SecretKeysOnDevice = secretKeys.IsOnDevice()
	config.CiphertextsOnDevice = ciphertexts.IsOnDevice()
	config.SharedSecretsOnDevice = sharedSecrets.IsOnDevice()

	if config.SecretKeysOnDevice {
		if dev, ok := secretKeys.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}
	if config.CiphertextsOnDevice {
		if dev, ok := ciphertexts.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}
	if config.SharedSecretsOnDevice {
		if dev, ok := sharedSecrets.(core.DeviceSlice); ok {
			dev.CheckDevice()
		} else {
			return runtime.InvalidArgument
		}
	}

	cSecretKeys := (*C.uint8_t)(secretKeys.AsUnsafePointer())
	cCiphertexts := (*C.uint8_t)(ciphertexts.AsUnsafePointer())
	cConfig := (*C.MlKemConfig)(unsafe.Pointer(&config))
	cSharedSecrets := (*C.uint8_t)(sharedSecrets.AsUnsafePointer())

	__ret := params.DecapsulateFFI(cSecretKeys, cCiphertexts, cConfig, cSharedSecrets)
	return runtime.EIcicleError(__ret)
}
