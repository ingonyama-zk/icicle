use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::{DeviceSlice, HostOrDeviceSlice};

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const CEXTENSION_LIMBS: usize = 2;
pub(crate) const QEXTENSION_LIMBS: usize = 4;

impl_scalar_field!("m31", m31, SCALAR_LIMBS, ScalarField, ScalarCfg, Fr);
impl_scalar_field!(
    "m31_cextension",
    m31_cextension,
    CEXTENSION_LIMBS,
    CExtensionField,
    CExtensionCfg,
    Fr
);
impl_scalar_field!(
    "m31_extension",
    m31_extension,
    QEXTENSION_LIMBS,
    ExtensionField,
    ExtensionCfg,
    Fr
);
#[cfg(test)]
mod tests {
    use super::{ExtensionField, ScalarField};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
    mod extension {
        use super::*;

        impl_field_tests!(ExtensionField);
    }
}
