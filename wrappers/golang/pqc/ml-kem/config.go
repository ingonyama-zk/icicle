package mlkem

import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"
)

type MlKemConfig struct {
	StreamHandle          runtime.Stream
	IsAsync               bool
	MessagesOnDevice      bool
	EntropyOnDevice       bool
	PublicKeysOnDevice    bool
	SecretKeysOnDevice    bool
	CiphertextsOnDevice   bool
	SharedSecretsOnDevice bool
	BatchSize             uint64
	Ext                   config_extension.ConfigExtensionHandler
}

func GetDefaultMlKemConfig() MlKemConfig {
	return MlKemConfig{
		StreamHandle:          nil,
		IsAsync:               false,
		MessagesOnDevice:      false,
		EntropyOnDevice:       false,
		PublicKeysOnDevice:    false,
		SecretKeysOnDevice:    false,
		CiphertextsOnDevice:   false,
		SharedSecretsOnDevice: false,
		BatchSize:             1,
		Ext:                   nil,
	}
}
