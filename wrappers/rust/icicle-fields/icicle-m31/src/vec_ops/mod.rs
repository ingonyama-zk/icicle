use crate::field::{
    ComplexExtensionCfg, ComplexExtensionField, QuarticExtensionCfg, QuarticExtensionField, ScalarCfg, ScalarField,
};

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("m31", m31, ScalarField, ScalarCfg);
impl_vec_ops_field!(
    "m31_complex_extension",
    m31_complex_extension,
    ComplexExtensionField,
    ComplexExtensionCfg
);
impl_vec_ops_field!(
    "m31_quartic_extension",
    m31_quartic_extension,
    QuarticExtensionField,
    QuarticExtensionCfg
);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ComplexExtensionField, QuarticExtensionField, ScalarField};
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(ScalarField);

    mod complex_extension {
        use super::*;

        impl_vec_ops_tests!(ComplexExtensionField);
    }

    mod quartic_extension {
        use super::*;

        impl_vec_ops_tests!(QuarticExtensionField);
    }
}
