use crate::field::{KoalabearExtensionField, KoalabearField};

use icicle_core::vec_ops::{MixedVecOps, VecOps, VecOpsConfig};
use icicle_core::{impl_vec_ops_field, impl_vec_ops_mixed_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

use icicle_core::program::Program;

impl_vec_ops_field!("koalabear", koalabear, KoalabearField);
impl_vec_ops_field!("koalabear_extension", koalabear_extension, KoalabearExtensionField);
impl_vec_ops_mixed_field!(
    "koalabear_extension",
    koalabear_mixed,
    KoalabearExtensionField,
    KoalabearField
);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{KoalabearExtensionField, KoalabearField};
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_mixed_vec_ops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(koalabear, KoalabearField);

    mod extension {
        use super::*;

        impl_vec_ops_tests!(koalabear_extension, KoalabearExtensionField);
        impl_mixed_vec_ops_tests!(KoalabearExtensionField, KoalabearField);
    }
}
