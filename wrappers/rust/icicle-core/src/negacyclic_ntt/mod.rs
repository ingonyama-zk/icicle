use crate::ntt::NTTDir;
use crate::polynomial_ring::PolynomialRing;
use icicle_runtime::config::ConfigExtension;
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStreamHandle;

pub mod tests;

/// Configuration for negacyclic NTT operations.
/// Used to control execution mode, memory location, and async behavior.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NegacyclicNttConfig {
    /// CUDA stream or CPU stream handle
    pub stream_handle: IcicleStreamHandle,
    /// Whether inputs reside in device memory
    pub are_inputs_on_device: bool,
    /// Whether outputs reside in device memory
    pub are_outputs_on_device: bool,
    /// Whether the operation should be executed asynchronously
    pub is_async: bool,
    /// Extension for backend-specific config values
    pub ext: ConfigExtension,
}

impl Default for NegacyclicNttConfig {
    fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

/// Trait defining negacyclic NTT operations for a given polynomial ring.
pub trait NegacyclicNtt<P: PolynomialRing> {
    /// Executes a negacyclic NTT (forward or inverse) between slices.
    fn ntt(
        input: &(impl HostOrDeviceSlice<P> + ?Sized),
        dir: NTTDir,
        cfg: &NegacyclicNttConfig,
        output: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    ) -> Result<(), eIcicleError>;

    /// Executes an in-place negacyclic NTT (forward or inverse).
    fn ntt_inplace(
        inout: &mut (impl HostOrDeviceSlice<P> + ?Sized),
        dir: NTTDir,
        cfg: &NegacyclicNttConfig,
    ) -> Result<(), eIcicleError>;
}

/// Floating wrapper functions
pub fn ntt<P: PolynomialRing + NegacyclicNtt<P>>(
    input: &(impl HostOrDeviceSlice<P> + ?Sized),
    dir: NTTDir,
    cfg: &NegacyclicNttConfig,
    output: &mut (impl HostOrDeviceSlice<P> + ?Sized),
) -> Result<(), eIcicleError> {
    P::ntt(input, dir, cfg, output)
}

pub fn ntt_inplace<P: PolynomialRing + NegacyclicNtt<P>>(
    inout: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    dir: NTTDir,
    cfg: &NegacyclicNttConfig,
) -> Result<(), eIcicleError> {
    P::ntt_inplace(inout, dir, cfg)
}

/// Macro to implement `NegacyclicNtt` trait and FFI binding for a concrete polynomial type.
#[macro_export]
macro_rules! impl_negacyclic_ntt {
    ($prefix:literal, $poly:ty, $base:ty) => {
        extern "C" {
            #[link_name = concat!($prefix, "_negacyclic_ntt")]
            fn ntt_ffi(
                input: *const $poly,
                size: i32,
                dir: NTTDir,
                cfg: &NegacyclicNttConfig,
                output: *mut $poly,
            ) -> eIcicleError;
        }

        impl NegacyclicNtt<$poly> for $poly {
            fn ntt(
                input: &(impl HostOrDeviceSlice<$poly> + ?Sized),
                dir: NTTDir,
                cfg: &NegacyclicNttConfig,
                output: &mut (impl HostOrDeviceSlice<$poly> + ?Sized),
            ) -> Result<(), eIcicleError> {
                if input.is_on_device() && !input.is_on_active_device() {
                    panic!("input not allocated on the active device");
                }
                if output.is_on_device() && !output.is_on_active_device() {
                    panic!("output not allocated on the active device");
                }

                let mut local_cfg = cfg.clone();
                local_cfg.are_inputs_on_device = input.is_on_device();
                local_cfg.are_outputs_on_device = output.is_on_device();

                unsafe {
                    ntt_ffi(
                        input.as_ptr(),
                        input.len() as i32,
                        dir,
                        &local_cfg,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn ntt_inplace(
                inout: &mut (impl HostOrDeviceSlice<$poly> + ?Sized),
                dir: NTTDir,
                cfg: &NegacyclicNttConfig,
            ) -> Result<(), eIcicleError> {
                if inout.is_on_device() && !inout.is_on_active_device() {
                    panic!("inout not allocated on the active device");
                }

                let mut local_cfg = cfg.clone();
                local_cfg.are_inputs_on_device = inout.is_on_device();
                local_cfg.are_outputs_on_device = inout.is_on_device();

                unsafe {
                    ntt_ffi(
                        inout.as_ptr(),
                        inout.len() as i32,
                        dir,
                        &local_cfg,
                        inout.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_negacyclic_ntt_tests {
    ($poly:ty) => {
        #[cfg(test)]
        mod negacyclic_ntt_tests {
            use super::*;
            use icicle_runtime::test_utilities;
            use $crate::negacyclic_ntt::tests::*;

            pub fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            // Phase2 due to ntt-domain conflict
            #[test]
            fn phase2_test_ntt_roundtrip() {
                initialize();
                test_negacyclic_ntt_roundtrip::<$poly>();
            }
        }
    };
}
