package mlkem

import (
	"fmt"
	"os"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func KeygenCheck(entropy, publicKeys, secretKeys core.HostOrDeviceSlice, config *MlKemConfig, params *KyberParams) runtime.EIcicleError {
	if uint64(entropy.Len()) != config.BatchSize*ENTROPY_BYTES {
		fmt.Fprintf(os.Stderr, "Keygen error: entropy length (%d) != expected (%d)\n", entropy.Len(), config.BatchSize*ENTROPY_BYTES)
		return runtime.InvalidArgument
	}
	if uint64(publicKeys.Len()) != config.BatchSize*uint64(params.PublicKeyBytes) {
		fmt.Fprintf(os.Stderr, "Keygen error: publicKeys length (%d) != expected (%d)\n", publicKeys.Len(), config.BatchSize*uint64(params.PublicKeyBytes))
		return runtime.InvalidArgument
	}
	if uint64(secretKeys.Len()) != config.BatchSize*uint64(params.SecretKeyBytes) {
		fmt.Fprintf(os.Stderr, "Keygen error: secretKeys length (%d) != expected (%d)\n", secretKeys.Len(), config.BatchSize*uint64(params.SecretKeyBytes))
		return runtime.InvalidArgument
	}

	config.EntropyOnDevice = entropy.IsOnDevice()
	config.PublicKeysOnDevice = publicKeys.IsOnDevice()
	config.SecretKeysOnDevice = secretKeys.IsOnDevice()

	if config.EntropyOnDevice {
		entropy.(core.DeviceSlice).CheckDevice()
	}

	if config.PublicKeysOnDevice {
		publicKeys.(core.DeviceSlice).CheckDevice()
	}

	if config.SecretKeysOnDevice {
		secretKeys.(core.DeviceSlice).CheckDevice()
	}

	return runtime.Success
}

func EncapsCheck(message, publicKeys, ciphertexts, sharedSecrets core.HostOrDeviceSlice, config *MlKemConfig, params *KyberParams) runtime.EIcicleError {
	if uint64(message.Len()) != config.BatchSize*uint64(MESSAGE_BYTES) {
		fmt.Fprintf(os.Stderr, "Encapsulate error: message length (%d) != expected (%d)\n", message.Len(), config.BatchSize*uint64(MESSAGE_BYTES))
		return runtime.InvalidArgument
	}
	if uint64(publicKeys.Len()) != config.BatchSize*uint64(params.PublicKeyBytes) {
		fmt.Fprintf(os.Stderr, "Encapsulate error: publicKeys length (%d) != expected (%d)\n", publicKeys.Len(), config.BatchSize*uint64(params.PublicKeyBytes))
		return runtime.InvalidArgument
	}
	if uint64(ciphertexts.Len()) != config.BatchSize*uint64(params.CiphertextBytes) {
		fmt.Fprintf(os.Stderr, "Encapsulate error: ciphertexts length (%d) != expected (%d)\n", ciphertexts.Len(), config.BatchSize*uint64(params.CiphertextBytes))
		return runtime.InvalidArgument
	}
	if uint64(sharedSecrets.Len()) != config.BatchSize*uint64(params.SharedSecretBytes) {
		fmt.Fprintf(os.Stderr, "Encapsulate error: sharedSecrets length (%d) != expected (%d)\n", sharedSecrets.Len(), config.BatchSize*uint64(params.SharedSecretBytes))
		return runtime.InvalidArgument
	}

	config.MessagesOnDevice = message.IsOnDevice()
	config.PublicKeysOnDevice = publicKeys.IsOnDevice()
	config.CiphertextsOnDevice = ciphertexts.IsOnDevice()
	config.SharedSecretsOnDevice = sharedSecrets.IsOnDevice()

	if config.MessagesOnDevice {
		message.(core.DeviceSlice).CheckDevice()
	}
	if config.PublicKeysOnDevice {
		publicKeys.(core.DeviceSlice).CheckDevice()
	}
	if config.CiphertextsOnDevice {
		ciphertexts.(core.DeviceSlice).CheckDevice()
	}
	if config.SharedSecretsOnDevice {
		sharedSecrets.(core.DeviceSlice).CheckDevice()
	}

	return runtime.Success
}

func DecapsCheck(secretKeys, ciphertexts, sharedSecrets core.HostOrDeviceSlice, config *MlKemConfig, params *KyberParams) runtime.EIcicleError {
	if uint64(secretKeys.Len()) != config.BatchSize*uint64(params.SecretKeyBytes) {
		fmt.Fprintf(os.Stderr, "Decapsulate error: secretKeys length (%d) != expected (%d)\n", secretKeys.Len(), config.BatchSize*uint64(params.SecretKeyBytes))
		return runtime.InvalidArgument
	}
	if uint64(ciphertexts.Len()) != config.BatchSize*uint64(params.CiphertextBytes) {
		fmt.Fprintf(os.Stderr, "Decapsulate error: ciphertexts length (%d) != expected (%d)\n", ciphertexts.Len(), config.BatchSize*uint64(params.CiphertextBytes))
		return runtime.InvalidArgument
	}
	if uint64(sharedSecrets.Len()) != config.BatchSize*uint64(params.SharedSecretBytes) {
		fmt.Fprintf(os.Stderr, "Decapsulate error: sharedSecrets length (%d) != expected (%d)\n", sharedSecrets.Len(), config.BatchSize*uint64(params.SharedSecretBytes))
		return runtime.InvalidArgument
	}

	config.SecretKeysOnDevice = secretKeys.IsOnDevice()
	config.CiphertextsOnDevice = ciphertexts.IsOnDevice()
	config.SharedSecretsOnDevice = sharedSecrets.IsOnDevice()

	if config.SecretKeysOnDevice {
		secretKeys.(core.DeviceSlice).CheckDevice()
	}
	if config.CiphertextsOnDevice {
		ciphertexts.(core.DeviceSlice).CheckDevice()
	}
	if config.SharedSecretsOnDevice {
		sharedSecrets.(core.DeviceSlice).CheckDevice()
	}

	return runtime.Success
}
