use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::{DeviceSlice, HostOrDeviceSlice};

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_scalar_field!("babybear", babybear, SCALAR_LIMBS, ScalarField, ScalarCfg, Fr);
impl_scalar_field!(
    "babybear_extension",
    babybear_extension,
    EXTENSION_LIMBS,
    QuarticExtensionField,
    ExtensionCfg,
    Fr
);

#[cfg(test)]
mod tests {
    use super::{QuarticExtensionField, ScalarField};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
    mod extension {
        use super::*;

        impl_field_tests!(QuarticExtensionField);
    }
}
