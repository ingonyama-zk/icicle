use crate::traits::FieldImpl;
use icicle_runtime::{
    config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, stream::{IcicleStreamHandle, IcicleStream},
};

pub mod tests;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct SumcheckConfig {
    pub stream_handle: IcicleStreamHandle,
    pub are_inputs_on_device: bool,
    pub are_outputs_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl SumcheckConfig {
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

#[doc(hidden)]
pub trait Sumcheck<T, F: FieldImpl> {
    fn sumcheck_unchecked(
        evals: &(impl HostOrDeviceSlice<T> + ?Sized),
        cubic_polys: &(impl HostOrDeviceSlice<T> + ?Sized),
        transcript: &(impl HostOrDeviceSlice<T> + ?Sized),
        c: &(impl HostOrDeviceSlice<T> + ?Sized),
        num_rounds: usize,
        nof_polys: usize,
        cfg: &IcicleStream,
    ) -> Result<(), eIcicleError>;
}

pub fn sumcheck<T, F>(
    evals: &(impl HostOrDeviceSlice<T> + ?Sized),
    cubic_polys: &(impl HostOrDeviceSlice<T> + ?Sized),
    transcript: &(impl HostOrDeviceSlice<T> + ?Sized),
    c: &(impl HostOrDeviceSlice<T> + ?Sized),
    num_rounds: usize,
    nof_polys: usize,
    cfg: &IcicleStream,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sumcheck<T, F>,
{
    <<F as FieldImpl>::Config as Sumcheck<T, F>>::sumcheck_unchecked(evals, cubic_polys, transcript, c, num_rounds, nof_polys, cfg)
}

#[macro_export]
macro_rules! impl_sumcheck_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
    ) => {
        mod $field_prefix_ident {

            use crate::sumcheck::{$field, HostOrDeviceSlice};
            use icicle_core::sumcheck::SumcheckConfig;
            use icicle_runtime::{stream::IcicleStream, errors::eIcicleError};

            extern "C" {
                #[link_name = concat!($field_prefix, "_sumcheck")]
                pub(crate) fn sumcheck_generic_unified(
                    evals: *const $field,
                    cubic_polys: *const $field,
                    transcript: *const $field,
                    c: *const $field,
                    num_rounds: usize,
                    nof_polys: usize,
                    stream: *const IcicleStream,
                ) -> eIcicleError;
            }
        }

        impl Sumcheck<$field, $field> for $field_config {
            fn sumcheck_unchecked(
                evals: &(impl HostOrDeviceSlice<$field> + ?Sized),
                cubic_polys: &(impl HostOrDeviceSlice<$field> + ?Sized),
                transcript: &(impl HostOrDeviceSlice<$field> + ?Sized),
                c: &(impl HostOrDeviceSlice<$field> + ?Sized),
                num_rounds: usize,
                nof_polys: usize,
                stream: &IcicleStream,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::sumcheck_generic_unified(
                        evals.as_ptr(),
                        cubic_polys.as_ptr(),
                        transcript.as_ptr(),
                        c.as_ptr(),
                        num_rounds,
                        nof_polys,
                        stream as *const IcicleStream,
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_sumcheck_tests {
    (
      $field:ident
    ) => {
        pub(crate) mod test_sumcheck {
            use super::*;
            use icicle_core::test_utilities;
            use icicle_runtime::{device::Device, runtime};
            use std::sync::Once;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
            }

            #[test]
            pub fn test_sumcheck_scalars() {
                initialize();
                check_sumcheck_scalars::<$field>()
            }
        }
    };
}
