use icicle_cuda_runtime::{
    device_context::{DeviceContext, DEFAULT_DEVICE_ID},
    memory::HostOrDeviceSlice,
};

use crate::{error::IcicleResult, traits::FieldImpl};

pub trait Fft<F: FieldImpl> {
    fn evaluate_unchecked(inout: &mut HostOrDeviceSlice<F>, ws: &mut HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>;
    fn interpolate_unchecked(
        inout: &mut HostOrDeviceSlice<F>,
        ws: &mut HostOrDeviceSlice<F>,
        n: u32,
    ) -> IcicleResult<()>;
}

pub fn fft_evaluate<F>(inout: &mut HostOrDeviceSlice<F>, ws: &mut HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Fft<F>,
{
    <<F as FieldImpl>::Config as Fft<F>>::evaluate_unchecked(inout, ws, n)
}

pub fn fft_interpolate<F>(inout: &mut HostOrDeviceSlice<F>, ws: &mut HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Fft<F>,
{
    <<F as FieldImpl>::Config as Fft<F>>::interpolate_unchecked(inout, ws, n)
}

#[macro_export]
macro_rules! impl_fft {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
      ) => {
        mod $field_prefix_ident {
            use crate::fft::{$field, $field_config, CudaError, DeviceContext};

            extern "C" {
                #[link_name = concat!($field_prefix, "FftEvaluate")]
                pub(crate) fn _fft_evaluate(inout: *mut $field, ws: *mut $field, n: u32) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "FftInterpolate")]
                pub(crate) fn _fft_interpolate(inout: *mut $field, ws: *mut $field, n: u32) -> CudaError;
            }
        }

        impl Fft<$field> for $field_config {
            fn evaluate_unchecked(
                inout: &mut HostOrDeviceSlice<$field>,
                ws: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_fft_evaluate(inout.as_mut_ptr(), ws.as_mut_ptr(), n).wrap() }
            }

            fn interpolate_unchecked(
                inout: &mut HostOrDeviceSlice<$field>,
                ws: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_fft_interpolate(inout.as_mut_ptr(), ws.as_mut_ptr(), n).wrap() }
            }
        }
    };
}
