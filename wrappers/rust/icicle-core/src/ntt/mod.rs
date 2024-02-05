use icicle_cuda_runtime::device_context::{get_default_device_context, DeviceContext};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::{error::IcicleResult, traits::FieldImpl};

#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;

/// Whether to perform normal forward NTT, or inverse NTT (iNTT). Mathematically, forward NTT computes polynomial
/// evaluations from coefficients while inverse NTT computes coefficients from evaluations.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NTTDir {
    kForward,
    kInverse,
}

/// How to order inputs and outputs of the NTT. If needed, use this field to specify decimation: decimation in time
/// (DIT) corresponds to `Ordering::kRN` while decimation in frequency (DIF) to `Ordering::kNR`. Also, to specify
/// butterfly to be used, select `Ordering::kRN` for Cooley-Tukey and `Ordering::kNR` for Gentleman-Sande. There's
/// no implication that a certain decimation or butterfly will actually be used under the hood, this is just for
/// compatibility with codebases that use "decimation" and "butterfly" to denote ordering of inputs and outputs.
///
/// Ordering options are:
/// - kNN: inputs and outputs are natural-order (example of natural ordering: `a_0, a_1, a_2, a_3, a_4, a_5, a_6,
/// a_7`.
/// - kNR: inputs are natural-order and outputs are bit-reversed-order (example of bit-reversed ordering: `a_0,
/// a_4, a_2, a_6, a_1, a_5, a_3, a_7`.
/// - kRN: inputs are bit-reversed-order and outputs are natural-order.
/// - kRR: inputs and outputs are bit-reversed-order.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    kNN,
    kNR,
    kRN,
    kRR,
}

/// Struct that encodes NTT parameters to be passed into the [ntt](ntt) function.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NTTConfig<'a, S> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
    /// Coset generator. Used to perform coset (i)NTTs. Default value: `S::one()` (corresponding to no coset being used).
    pub coset_gen: S,
    /// The number of NTTs to compute. Default value: 1.
    pub batch_size: i32,
    /// Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`.
    pub ordering: Ordering,
    are_inputs_on_device: bool,
    are_outputs_on_device: bool,
    /// Whether to run the NTT asynchronously. If set to `true`, the NTT function will be non-blocking and you'd need to synchronize
    /// it explicitly by running `stream.synchronize()`. If set to false, the NTT function will block the current CPU thread.
    pub is_async: bool,
}

impl<'a, S: FieldImpl> NTTConfig<'a, S> {
    pub fn default_config() -> Self {
        let ctx = get_default_device_context();
        NTTConfig {
            ctx,
            coset_gen: S::one(),
            batch_size: 1,
            ordering: Ordering::kNN,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
        }
    }
}

#[doc(hidden)]
pub trait NTT<F: FieldImpl> {
    fn ntt_unchecked(
        input: &HostOrDeviceSlice<F>,
        dir: NTTDir,
        cfg: &NTTConfig<F>,
        output: &mut HostOrDeviceSlice<F>,
    ) -> IcicleResult<()>;
    fn initialize_domain(primitive_root: F, ctx: &DeviceContext) -> IcicleResult<()>;
    fn get_default_ntt_config() -> NTTConfig<'static, F>;
}

/// Computes the NTT, or a batch of several NTTs.
///
/// # Arguments
///
/// * `input` - inputs of the NTT.
///
/// * `dir` - whether to compute forward of inverse NTT.
///
/// * `cfg` - config used to specify extra arguments of the NTT.
///
/// * `output` - buffer to write the NTT outputs into. Must be of the same size as `input`.
pub fn ntt<F>(
    input: &HostOrDeviceSlice<F>,
    dir: NTTDir,
    cfg: &NTTConfig<F>,
    output: &mut HostOrDeviceSlice<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>,
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

    <<F as FieldImpl>::Config as NTT<F>>::ntt_unchecked(input, dir, &local_cfg, output)
}

/// Generates twiddle factors which will be used to compute NTTs.
///
/// # Arguments
///
/// * `primitive_root` - primitive root to generate twiddles from. Should be of large enough order to cover all
/// NTTs that you need. For example, if NTTs of sizes 2^17 and 2^18 are computed, use the primitive root of order 2^18.
/// This function will panic if the order of `primitive_root` is not a power of two.
///
/// * `ctx` - GPU index and stream to perform the computation.
pub fn initialize_domain<F>(primitive_root: F, ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>,
{
    <<F as FieldImpl>::Config as NTT<F>>::initialize_domain(primitive_root, ctx)
}

/// Returns [NTT config](NTTConfig) struct populated with default values.
pub fn get_default_ntt_config<F>() -> NTTConfig<'static, F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>,
{
    <<F as FieldImpl>::Config as NTT<F>>::get_default_ntt_config()
}

#[macro_export]
macro_rules! impl_ntt {
    (
      $field_prefix:literal,
      $field_prefix_ident:ident,
      $field:ident,
      $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use crate::ntt::{$field, $field_config, CudaError, DeviceContext, NTTConfig, NTTDir};

            extern "C" {
                #[link_name = concat!($field_prefix, "NTTCuda")]
                pub(crate) fn ntt_cuda(
                    input: *const $field,
                    size: i32,
                    dir: NTTDir,
                    config: &NTTConfig<$field>,
                    output: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "InitializeDomain")]
                pub(crate) fn initialize_ntt_domain(primitive_root: $field, ctx: &DeviceContext) -> CudaError;
            }
        }

        impl NTT<$field> for $field_config {
            fn ntt_unchecked(
                input: &HostOrDeviceSlice<$field>,
                dir: NTTDir,
                cfg: &NTTConfig<$field>,
                output: &mut HostOrDeviceSlice<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::ntt_cuda(
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
                unsafe { $field_prefix_ident::initialize_ntt_domain(primitive_root, ctx).wrap() }
            }

            fn get_default_ntt_config() -> NTTConfig<'static, $field> {
                NTTConfig::<$field>::default_config()
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ntt_tests {
    (
      $field:ident
    ) => {
        const MAX_SIZE: u64 = 1 << 17;
        static INIT: OnceLock<()> = OnceLock::new();

        #[test]
        fn test_ntt() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE));
            check_ntt::<$field>()
        }

        #[test]
        fn test_ntt_coset_from_subgroup() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE));
            check_ntt_coset_from_subgroup::<$field>()
        }

        #[test]
        fn test_ntt_arbitrary_coset() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE));
            check_ntt_arbitrary_coset::<$field>()
        }

        #[test]
        fn test_ntt_batch() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE));
            check_ntt_batch::<$field>()
        }

        #[test]
        fn test_ntt_device_async() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE));
            check_ntt_device_async::<$field>()
        }
    };
}
