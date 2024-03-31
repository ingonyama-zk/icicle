use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::{
    curve::Curve,
    ntt::{FieldImpl, IcicleResult, NTTConfig, NTTDir},
};

pub use crate::curve::Projective;

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
    // fn initialize_domain(primitive_root: C::ScalarField, ctx: &DeviceContext) -> IcicleResult<()>;
    // fn initialize_domain_fast_twiddles_mode(primitive_root: C::ScalarField, ctx: &DeviceContext) -> IcicleResult<()>;
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
    <C::BaseField as FieldImpl>::Config: ECNTT<C>,
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
    <C::BaseField as FieldImpl>::Config::ecntt_unchecked(input, dir, &local_cfg, output)
}

#[macro_export]
macro_rules! impl_ecntt {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident,
        $base_field:ident,
        $curve:ident
    ) => {
        mod $field_prefix_ident {
            use crate::ecntt::Projective;
            use crate::ecntt::{
                $field, $field_config, $curve, $base_field, CudaError, DeviceContext, NTTConfig, NTTDir, DEFAULT_DEVICE_ID,
            };

            extern "C" {
                #[link_name = concat!($field_prefix, "ECNTTCuda")]
                pub(crate) fn ecntt_cuda(
                    input: *const Projective<$curve>,
                    size: i32,
                    dir: NTTDir,
                    config: &NTTConfig<$field>,
                    output: *mut Projective<$curve>,
                ) -> CudaError;
            }
        }
        //<C::ScalarField as FieldImpl>::Config
        impl ECNTT<$curve> for $base_field {
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
        }
    };
}

#[macro_export]
macro_rules! impl_ecntt_tests {
    (
      $field:ident,
      $base_field:ident,
      $curve:ident
    ) => {
        use icicle_core::ntt::tests::init_domain;
        use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;
        const MAX_SIZE: u64 = 1 << 18;
        static INIT: OnceLock<()> = OnceLock::new();
        const FAST_TWIDDLES_MODE: bool = false;

        #[test]
        fn test_ecntt() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ecntt::<$field, $base_field, $curve>()
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
