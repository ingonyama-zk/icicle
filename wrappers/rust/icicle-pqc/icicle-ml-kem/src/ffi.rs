use crate::config::MlKemConfig;
use icicle_runtime::eIcicleError;

extern "C" {
    #[link_name = "icicle_ml_kem_keygen512"]
    pub(crate) fn keygen_ffi512(
        entropy: *const u8,
        config: *const MlKemConfig,
        public_keys: *mut u8,
        secret_keys: *mut u8,
    ) -> eIcicleError;

    #[link_name = "icicle_ml_kem_encapsulate512"]
    pub(crate) fn encapsulate_ffi512(
        message: *const u8,
        public_keys: *const u8,
        config: *const MlKemConfig,
        ciphertexts: *mut u8,
        shared_secrets: *mut u8,
    ) -> eIcicleError;

    #[link_name = "icicle_ml_kem_decapsulate512"]
    pub(crate) fn decapsulate_ffi512(
        secret_keys: *const u8,
        ciphertexts: *const u8,
        config: *const MlKemConfig,
        shared_secrets: *mut u8,
    ) -> eIcicleError;

    #[link_name = "icicle_ml_kem_keygen768"]
    pub(crate) fn keygen_ffi768(
        entropy: *const u8,
        config: *const MlKemConfig,
        public_keys: *mut u8,
        secret_keys: *mut u8,
    ) -> eIcicleError;

    #[link_name = "icicle_ml_kem_encapsulate768"]
    pub(crate) fn encapsulate_ffi768(
        message: *const u8,
        public_keys: *const u8,
        config: *const MlKemConfig,
        ciphertexts: *mut u8,
        shared_secrets: *mut u8,
    ) -> eIcicleError;

    #[link_name = "icicle_ml_kem_decapsulate768"]
    pub(crate) fn decapsulate_ffi768(
        secret_keys: *const u8,
        ciphertexts: *const u8,
        config: *const MlKemConfig,
        shared_secrets: *mut u8,
    ) -> eIcicleError;

    #[link_name = "icicle_ml_kem_keygen1024"]
    pub(crate) fn keygen_ffi1024(
        entropy: *const u8,
        config: *const MlKemConfig,
        public_keys: *mut u8,
        secret_keys: *mut u8,
    ) -> eIcicleError;

    #[link_name = "icicle_ml_kem_encapsulate1024"]
    pub(crate) fn encapsulate_ffi1024(
        message: *const u8,
        public_keys: *const u8,
        config: *const MlKemConfig,
        ciphertexts: *mut u8,
        shared_secrets: *mut u8,
    ) -> eIcicleError;

    #[link_name = "icicle_ml_kem_decapsulate1024"]
    pub(crate) fn decapsulate_ffi1024(
        secret_keys: *const u8,
        ciphertexts: *const u8,
        config: *const MlKemConfig,
        shared_secrets: *mut u8,
    ) -> eIcicleError;
}
