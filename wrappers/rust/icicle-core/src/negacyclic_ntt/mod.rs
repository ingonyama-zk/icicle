use crate::ntt::NTTDir;
use crate::polynomial_ring::PolynomialRing;
use icicle_runtime::config::ConfigExtension;
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStreamHandle;

pub mod tests;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct NegacyclicNttConfig {
    pub stream_handle: IcicleStreamHandle,
    pub are_inputs_on_device: bool,
    pub are_outputs_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl NegacyclicNttConfig {
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

pub trait NegacyclicNtt<P: PolynomialRing> {
    fn ntt(
        input: &(impl HostOrDeviceSlice<P> + ?Sized),
        dir: NTTDir,
        cfg: &NegacyclicNttConfig,
        output: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    ) -> Result<(), eIcicleError>;

    fn ntt_inplace(
        inout: &mut (impl HostOrDeviceSlice<P> + ?Sized),
        dir: NTTDir,
        cfg: &NegacyclicNttConfig,
    ) -> Result<(), eIcicleError>;
}

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
                unsafe { ntt_ffi(input.as_ptr(), input.len() as i32, dir, cfg, output.as_mut_ptr()).wrap() }
            }

            fn ntt_inplace(
                inout: &mut (impl HostOrDeviceSlice<$poly> + ?Sized),
                dir: NTTDir,
                cfg: &NegacyclicNttConfig,
            ) -> Result<(), eIcicleError> {
                unsafe { ntt_ffi(inout.as_ptr(), inout.len() as i32, dir, cfg, inout.as_mut_ptr()).wrap() }
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
            use $crate::negacyclic_ntt::tests::*;

            #[test]
            fn test_ntt_roundtrip() {
                test_negacyclic_ntt_roundtrip::<$poly>();
            }
        }
    };
}
