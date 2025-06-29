use crate::{traits::FieldImpl, vec_ops::VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

/// Norm type enum for specifying which norm to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    L2,        // Euclidean norm: sqrt(sum of squares)
    LInfinity, // Max norm: maximum absolute element value
}

/// Norm API for field/ring types.
///
/// This trait provides functionality to check if the norm of a vector is within a specified bound
/// or to compare norms of two vectors with a scaling factor.
pub trait Norm<T: FieldImpl> {
    /// Checks whether the norm of a vector is within a specified bound.
    ///
    /// This function assumes that:
    /// - Each element in the input vector is at most sqrt(q) in magnitude
    /// - The vector size is at most 2^16 elements
    fn check_norm_bound(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        norm_type: NormType,
        norm_bound: u64,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<bool> + ?Sized),
    ) -> Result<(), eIcicleError>;

    /// Checks whether norm(a) < scalar * norm(b)
    ///
    /// This is useful in lattice-based schemes and other relative-norm comparisons
    /// where an exact bound is not known in advance but depends on a second input vector.
    fn check_norm_relative(
        input_a: &(impl HostOrDeviceSlice<T> + ?Sized),
        input_b: &(impl HostOrDeviceSlice<T> + ?Sized),
        norm_type: NormType,
        scale: u64,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<bool> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

// Public floating functions around the trait
pub fn check_norm_bound<T: FieldImpl>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    norm_type: NormType,
    norm_bound: u64,
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<bool> + ?Sized),
) -> Result<(), eIcicleError>
where
    T::Config: Norm<T>,
{
    T::Config::check_norm_bound(input, norm_type, norm_bound, cfg, output)
}

pub fn check_norm_relative<T: FieldImpl>(
    input_a: &(impl HostOrDeviceSlice<T> + ?Sized),
    input_b: &(impl HostOrDeviceSlice<T> + ?Sized),
    norm_type: NormType,
    scale: u64,
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<bool> + ?Sized),
) -> Result<(), eIcicleError>
where
    T::Config: Norm<T>,
{
    T::Config::check_norm_relative(input_a, input_b, norm_type, scale, cfg, output)
}

/// Internal macro to implement the `Norm` trait for a specific field backend.
#[macro_export]
macro_rules! impl_norm {
    (
        $field_prefix: literal,
        $field_type: ident,
        $field_cfg_type: ident
    ) => {
        use icicle_core::norm::Norm;

        extern "C" {
            #[link_name = concat!($field_prefix, "_check_norm_bound")]
            fn check_norm_bound(
                input: *const $field_type,
                size: usize,
                norm_type: u32,
                norm_bound: u64,
                cfg: *const VecOpsConfig,
                output: *mut bool,
            ) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_check_norm_relative")]
            fn check_norm_relative(
                input_a: *const $field_type,
                input_b: *const $field_type,
                size: usize,
                norm_type: u32,
                scale: u64,
                cfg: *const VecOpsConfig,
                output: *mut bool,
            ) -> eIcicleError;
        }

        fn norm_check_args(
            input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
            cfg: &mut VecOpsConfig,
        ) -> Result<(), eIcicleError> {
            if input.len() % (cfg.batch_size as usize) != 0 {
                eprintln!(
                    "Batch size {} must divide input size {}",
                    cfg.batch_size,
                    input.len()
                );
                return Err(eIcicleError::InvalidArgument);
            }

            if input.is_on_device() && !input.is_on_active_device() {
                eprintln!("Input is on an inactive device");
                return Err(eIcicleError::InvalidArgument);
            }

            cfg.is_a_on_device = input.is_on_device();
            cfg.is_result_on_device = false; // Output is always on host

            Ok(())
        }

        impl Norm<$field_type> for $field_cfg_type {
            fn check_norm_bound(
                input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                norm_type: NormType,
                norm_bound: u64,
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<bool> + ?Sized),
            ) -> Result<(), eIcicleError> {
                let mut cfg = cfg.clone();
                norm_check_args(input, &mut cfg)?;

                unsafe {
                    check_norm_bound(
                        input.as_ptr(),
                        input.len() / (cfg.batch_size as usize),
                        norm_type as u32,
                        norm_bound,
                        &cfg,
                        output.as_mut_ptr(),
                    )
                    .wrap()?;
                }
                Ok(())
            }

            fn check_norm_relative(
                input_a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                input_b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                norm_type: NormType,
                scale: u64,
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<bool> + ?Sized),
            ) -> Result<(), eIcicleError> {
                let mut cfg = cfg.clone();
                norm_check_args(input_a, &mut cfg)?;
                norm_check_args(input_b, &mut cfg)?;

                if input_a.len() != input_b.len() {
                    return Err(eIcicleError::InvalidArgument);
                }

                unsafe {
                    check_norm_relative(
                        input_a.as_ptr(),
                        input_b.as_ptr(),
                        input_a.len() / (cfg.batch_size as usize),
                        norm_type as u32,
                        scale,
                        &cfg,
                        output.as_mut_ptr(),
                    )
                    .wrap()?;
                }
                Ok(())
            }
        }
    };
}

#[macro_export]
macro_rules! impl_norm_tests {
    ($field_type: ident) => {
        use icicle_core::norm::tests::*;
        use icicle_runtime::test_utilities;

        /// Initializes devices before running tests.
        pub fn initialize() {
            test_utilities::test_load_and_init_devices();
            test_utilities::test_set_main_device();
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn test_norm() {
                initialize();
                test_utilities::test_set_main_device();
                check_norm::<$field_type>();
                test_utilities::test_set_ref_device();
                check_norm::<$field_type>();
            }
        }
    };
}
