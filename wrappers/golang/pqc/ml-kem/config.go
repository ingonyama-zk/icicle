package mlkem

import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"
)

type MlKemConfig struct {
	StreamHandle          runtime.Stream
	IsAsync               bool
	messagesOnDevice      bool
	entropyOnDevice       bool
	publicKeysOnDevice    bool
	secretKeysOnDevice    bool
	ciphertextsOnDevice   bool
	sharedSecretsOnDevice bool
	BatchSize             uint64
	Ext                   config_extension.ConfigExtensionHandler
}

func GetDefaultMlKemConfig() MlKemConfig {
	return MlKemConfig{
		StreamHandle:          nil,
		IsAsync:               false,
		messagesOnDevice:      false,
		entropyOnDevice:       false,
		publicKeysOnDevice:    false,
		secretKeysOnDevice:    false,
		ciphertextsOnDevice:   false,
		sharedSecretsOnDevice: false,
		BatchSize:             1,
		Ext:                   nil,
	}
}
