use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::curve::{Curve, Projective};
use crate::ntt::{NTTConfig, NTTDir};
use crate::traits::FieldImpl;
use crate::error::IcicleResult;

#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;

#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;

#[doc(hidden)]
pub trait ECNTT<C: Curve> {
    fn ecntt_unchecked(
        input: &HostOrDeviceSlice<Projective<C>>,
        dir: NTTDir,
        cfg: &NTTConfig<C::ScalarField>,
        output: &mut HostOrDeviceSlice<Projective<C>>,
    ) -> IcicleResult<()>;
    fn initialize_domain(primitive_root: C::ScalarField, ctx: &DeviceContext) -> IcicleResult<()>;
    fn initialize_domain_fast_twiddles_mode(primitive_root: C::ScalarField, ctx: &DeviceContext) -> IcicleResult<()>;
}

/// Computes the ECNTT, or a batch of several ECNTTs.
///
/// # Arguments
///
/// * `input` - inputs of the ECNTT.
///
/// * `dir` - whether to compute forward of inverse ECNTT.
///
/// * `cfg` - config used to specify extra arguments of the ECNTT.
///
/// * `output` - buffer to write the ECNTT outputs into. Must be of the same size as `input`.
pub fn ecntt<C: Curve>(
    input: &HostOrDeviceSlice<Projective<C>>,
    dir: NTTDir,
    cfg: &NTTConfig<C::ScalarField>,
    output: &mut HostOrDeviceSlice<Projective<C>>,
) -> IcicleResult<()>
where
<C::ScalarField as FieldImpl>::Config: ECNTT<C>,
{
    if input.len() != output.len() {
        panic!(
            "input and output lengths {}; {} do not match",
            input.len(),
            output.len()
        );
    }
    let mut local_cfg = cfg.clone();
    local_cfg.are_inputs_on_device = input.is_on_device();
    local_cfg.are_outputs_on_device = output.is_on_device();
    //<<F as FieldImpl>::Config as NTT<F>>::ntt_unchecked(input, dir, &local_cfg, output)
    // <<C as Curve>::ArkSWConfig as ECNTT<C>>::ecntt_unchecked(input, dir, &local_cfg, output)
    <C::ScalarField as FieldImpl>::Config::ecntt_unchecked(input, dir, &local_cfg, output)
}

#[macro_export]
macro_rules! impl_ecntt {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident,
//
        // $curve_prefix:literal,
        // $curve_prefix_indent:ident,
        $curve:ident
    ) => {
        mod $field_prefix_ident {
            use crate::ntt::{$field, $field_config, CudaError, DeviceContext, NTTConfig, NTTDir, DEFAULT_DEVICE_ID};

            extern "C" {
                #[link_name = concat!($field_prefix, "ECNTTCuda")]
                pub(crate) fn ecntt_cuda(
                    input: *const $field,
                    size: i32,
                    dir: NTTDir,
                    config: &NTTConfig<$field>,
                    output: *mut $field,
                ) -> CudaError;
            }
        }
        //<C::ScalarField as FieldImpl>::Config
        impl ECNTT<$curve> for $field_config {
            fn ecntt_unchecked(
                input: &HostOrDeviceSlice<Projective<$curve>>,
                dir: NTTDir,
                cfg: &NTTConfig<$field>,
                output: &mut HostOrDeviceSlice<Projective<$curve>>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::ecntt_cuda(
                        input.as_ptr(),
                        (input.len() / (cfg.batch_size as usize)) as i32,
                        dir,
                        cfg,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn initialize_domain(primitive_root: $field, ctx: &DeviceContext) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::initialize_ntt_domain(&primitive_root, ctx, false).wrap() }
            }
            fn initialize_domain_fast_twiddles_mode(primitive_root: $field, ctx: &DeviceContext) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::initialize_ntt_domain(&primitive_root, ctx, true).wrap() }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ecntt_tests {
    (
      $field:ident,
      $curve_prefix:literal,
      $curve_prefix_indent:ident,
      $curve:ident

    ) => {
        const MAX_SIZE: u64 = 1 << 18;
        static INIT: OnceLock<()> = OnceLock::new();
        const FAST_TWIDDLES_MODE: bool = false;

        #[test]
        fn test_ecntt() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ecntt::<$field, $curve:ident>()
        }

        // #[test]
        // fn test_ntt_coset_from_subgroup() {
        //     INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
        //     check_ntt_coset_from_subgroup::<$field>()
        // }

        // #[test]
        // fn test_ntt_arbitrary_coset() {
        //     INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
        //     check_ntt_arbitrary_coset::<$field>()
        // }

        // #[test]
        // fn test_ntt_batch() {
        //     INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
        //     check_ntt_batch::<$field>()
        // }

        // #[test]
        // fn test_ntt_device_async() {
        //     // init_domain is in this test is performed per-device
        //     check_ntt_device_async::<$field>()
        // }
    };
}
