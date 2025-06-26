package mlkem

import (
	"fmt"
	"os"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func KeygenCheck(entropy, publicKeys, secretKeys core.HostOrDeviceSlice, config *MlKemConfig, params KyberMode) runtime.EIcicleError {
	if uint64(entropy.Len()) != config.BatchSize*ENTROPY_BYTES {
		fmt.Fprintf(os.Stderr, "Keygen error: entropy length (%d) != expected (%d)\n", entropy.Len(), config.BatchSize*ENTROPY_BYTES)
		return runtime.InvalidArgument
	}
	if uint64(publicKeys.Len()) != config.BatchSize*uint64(params.GetPublicKeyBytes()) {
		fmt.Fprintf(os.Stderr, "Keygen error: publicKeys length (%d) != expected (%d)\n", publicKeys.Len(), config.BatchSize*uint64(params.GetPublicKeyBytes()))
		return runtime.InvalidArgument
	}
	if uint64(secretKeys.Len()) != config.BatchSize*uint64(params.GetSecretKeyBytes()) {
		fmt.Fprintf(os.Stderr, "Keygen error: secretKeys length (%d) != expected (%d)\n", secretKeys.Len(), config.BatchSize*uint64(params.GetSecretKeyBytes()))
		return runtime.InvalidArgument
	}

	config.entropyOnDevice = entropy.IsOnDevice()
	config.publicKeysOnDevice = publicKeys.IsOnDevice()
	config.secretKeysOnDevice = secretKeys.IsOnDevice()

	if config.entropyOnDevice {
		entropy.(core.DeviceSlice).CheckDevice()
	}

	if config.publicKeysOnDevice {
		publicKeys.(core.DeviceSlice).CheckDevice()
	}

	if config.secretKeysOnDevice {
		secretKeys.(core.DeviceSlice).CheckDevice()
	}

	return runtime.Success
}

func EncapsCheck(message, publicKeys, ciphertexts, sharedSecrets core.HostOrDeviceSlice, config *MlKemConfig, params KyberMode) runtime.EIcicleError {
	if uint64(message.Len()) != config.BatchSize*uint64(MESSAGE_BYTES) {
		fmt.Fprintf(os.Stderr, "Encapsulate error: message length (%d) != expected (%d)\n", message.Len(), config.BatchSize*uint64(MESSAGE_BYTES))
		return runtime.InvalidArgument
	}
	if uint64(publicKeys.Len()) != config.BatchSize*uint64(params.GetPublicKeyBytes()) {
		fmt.Fprintf(os.Stderr, "Encapsulate error: publicKeys length (%d) != expected (%d)\n", publicKeys.Len(), config.BatchSize*uint64(params.GetPublicKeyBytes()))
		return runtime.InvalidArgument
	}
	if uint64(ciphertexts.Len()) != config.BatchSize*uint64(params.GetCiphertextBytes()) {
		fmt.Fprintf(os.Stderr, "Encapsulate error: ciphertexts length (%d) != expected (%d)\n", ciphertexts.Len(), config.BatchSize*uint64(params.GetCiphertextBytes()))
		return runtime.InvalidArgument
	}
	if uint64(sharedSecrets.Len()) != config.BatchSize*uint64(params.GetSharedSecretBytes()) {
		fmt.Fprintf(os.Stderr, "Encapsulate error: sharedSecrets length (%d) != expected (%d)\n", sharedSecrets.Len(), config.BatchSize*uint64(params.GetSharedSecretBytes()))
		return runtime.InvalidArgument
	}

	config.messagesOnDevice = message.IsOnDevice()
	config.publicKeysOnDevice = publicKeys.IsOnDevice()
	config.ciphertextsOnDevice = ciphertexts.IsOnDevice()
	config.sharedSecretsOnDevice = sharedSecrets.IsOnDevice()

	if config.messagesOnDevice {
		message.(core.DeviceSlice).CheckDevice()
	}
	if config.publicKeysOnDevice {
		publicKeys.(core.DeviceSlice).CheckDevice()
	}
	if config.ciphertextsOnDevice {
		ciphertexts.(core.DeviceSlice).CheckDevice()
	}
	if config.sharedSecretsOnDevice {
		sharedSecrets.(core.DeviceSlice).CheckDevice()
	}

	return runtime.Success
}

func DecapsCheck(secretKeys, ciphertexts, sharedSecrets core.HostOrDeviceSlice, config *MlKemConfig, params KyberMode) runtime.EIcicleError {
	if uint64(secretKeys.Len()) != config.BatchSize*uint64(params.GetSecretKeyBytes()) {
		fmt.Fprintf(os.Stderr, "Decapsulate error: secretKeys length (%d) != expected (%d)\n", secretKeys.Len(), config.BatchSize*uint64(params.GetSecretKeyBytes()))
		return runtime.InvalidArgument
	}
	if uint64(ciphertexts.Len()) != config.BatchSize*uint64(params.GetCiphertextBytes()) {
		fmt.Fprintf(os.Stderr, "Decapsulate error: ciphertexts length (%d) != expected (%d)\n", ciphertexts.Len(), config.BatchSize*uint64(params.GetCiphertextBytes()))
		return runtime.InvalidArgument
	}
	if uint64(sharedSecrets.Len()) != config.BatchSize*uint64(params.GetSharedSecretBytes()) {
		fmt.Fprintf(os.Stderr, "Decapsulate error: sharedSecrets length (%d) != expected (%d)\n", sharedSecrets.Len(), config.BatchSize*uint64(params.GetSharedSecretBytes()))
		return runtime.InvalidArgument
	}

	config.secretKeysOnDevice = secretKeys.IsOnDevice()
	config.ciphertextsOnDevice = ciphertexts.IsOnDevice()
	config.sharedSecretsOnDevice = sharedSecrets.IsOnDevice()

	if config.secretKeysOnDevice {
		secretKeys.(core.DeviceSlice).CheckDevice()
	}
	if config.ciphertextsOnDevice {
		ciphertexts.(core.DeviceSlice).CheckDevice()
	}
	if config.sharedSecretsOnDevice {
		sharedSecrets.(core.DeviceSlice).CheckDevice()
	}

	return runtime.Success
}
