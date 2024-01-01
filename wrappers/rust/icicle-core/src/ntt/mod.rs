use icicle_cuda_runtime::device_context::DeviceContext;

use crate::{error::IcicleResult, traits::FieldImpl};

#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;

/**
 * @enum NTTDir
 * Whether to perform normal forward NTT, or inverse NTT (iNTT). Mathematically, forward NTT computes polynomial
 * evaluations from coefficients while inverse NTT computes coefficients from evaluations.
 */
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NTTDir {
    kForward,
    kInverse,
}

/**
 * @enum Ordering
 * How to order inputs and outputs of the NTT. If needed, use this field to specify decimation: decimation in time
 * (DIT) corresponds to `Ordering::kRN` while decimation in frequency (DIF) to `Ordering::kNR`. Also, to specify
 * butterfly to be used, select `Ordering::kRN` for Cooley-Tukey and `Ordering::kNR` for Gentleman-Sande. There's
 * no implication that a certain decimation or butterfly will actually be used under the hood, this is just for
 * compatibility with codebases that use "decimation" and "butterfly" to denote ordering of inputs and outputs.
 *
 * Ordering options are:
 * - kNN: inputs and outputs are natural-order (example of natural ordering: \f$ \{a_0, a_1, a_2, a_3, a_4, a_5, a_6,
 * a_7\} \f$).
 * - kNR: inputs are natural-order and outputs are bit-reversed-order (example of bit-reversed ordering: \f$ \{a_0,
 * a_4, a_2, a_6, a_1, a_5, a_3, a_7\} \f$).
 * - kRN: inputs are bit-reversed-order and outputs are natural-order.
 * - kRR: inputs and outputs are bit-reversed-order.
 */
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    kNN,
    kNR,
    kRN,
    kRR,
}

/**
 * @struct NTTConfig
 * Struct that encodes NTT parameters to be passed into the [ntt](@ref ntt) function.
 */
#[repr(C)]
#[derive(Debug)]
pub struct NTTConfig<'a, S> {
    /** Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext). */
    pub ctx: DeviceContext<'a>,
    /** Coset generator. Used to perform coset (i)NTTs. Default value: `S::one()` (corresponding to no coset being used). */
    pub coset_gen: S,
    /** The number of NTTs to compute. Default value: 1. */
    pub batch_size: i32,
    /** Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`. */
    pub ordering: Ordering,
    /** True if inputs are on device and false if they're on host. Default value: false. */
    pub are_inputs_on_device: bool,
    /** If true, output is preserved on device for subsequent use in config and not freed after calculation. Default value: false. */
    pub are_outputs_on_device: bool,
    /** Whether to run the NTT asyncronously. If set to `true`, the NTT function will be non-blocking and you'd need to synchronize
     *  it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the NTT
     *  function will block the current CPU thread. */
    pub is_async: bool,
}

#[doc(hidden)]
pub trait NTT<F: FieldImpl> {
    fn ntt(input: &[F], dir: NTTDir, cfg: &NTTConfig<F>, output: &mut [F]) -> IcicleResult<()>;
    fn initialize_domain(primitive_root: F, ctx: &DeviceContext) -> IcicleResult<()>;
    fn get_default_ntt_config() -> NTTConfig<'static, F>;
}

pub fn ntt<F>(input: &[F], dir: NTTDir, cfg: &NTTConfig<F>, output: &mut [F]) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>,
{
    <<F as FieldImpl>::Config as NTT<F>>::ntt(input, dir, cfg, output)
}

pub fn initialize_domain<F>(primitive_root: F, ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>,
{
    <<F as FieldImpl>::Config as NTT<F>>::initialize_domain(primitive_root, ctx)
}

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
      $field:ident,
      $field_config:ident
    ) => {
        extern "C" {
            #[link_name = concat!($field_prefix, "NTTCuda")]
            fn ntt_cuda(
                input: *const $field,
                size: i32,
                dir: NTTDir,
                config: &NTTConfig<$field>,
                output: *mut $field,
            ) -> CudaError;

            #[link_name = concat!($field_prefix, "DefaultNTTConfig")]
            fn default_ntt_config() -> NTTConfig<'static, $field>;

            #[link_name = concat!($field_prefix, "InitializeDomain")]
            fn initialize_ntt_domain(primitive_root: $field, ctx: &DeviceContext) -> CudaError;
        }

        impl NTT<$field> for $field_config {
            fn ntt(input: &[$field], dir: NTTDir, cfg: &NTTConfig<$field>, output: &mut [$field]) -> IcicleResult<()> {
                if input.len() != output.len() {
                    panic!("input and output lengths do not match")
                }

                //TODO: more validations for cfg

                unsafe {
                    ntt_cuda(
                        input as *const _ as *const $field,
                        (input.len() / (cfg.batch_size as usize)) as i32,
                        dir,
                        cfg,
                        output as *mut _ as *mut $field,
                    )
                    .wrap()
                }
            }

            fn initialize_domain(primitive_root: $field, ctx: &DeviceContext) -> IcicleResult<()> {
                unsafe { initialize_ntt_domain(primitive_root, ctx).wrap() }
            }

            fn get_default_ntt_config() -> NTTConfig<'static, $field> {
                unsafe { default_ntt_config() }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ntt_tests {
    (
      $field:ident
    ) => {
        const MAX_SIZE: u64 = 1 << 16;
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
