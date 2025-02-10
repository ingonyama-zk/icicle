use icicle_core::hash::{Hasher, HasherHandle};
use icicle_runtime::{config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, IcicleStreamHandle};

const PADDING_SIZE: u32 = 24;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PowConfig {
    pub stream: IcicleStreamHandle, // `icicleStreamHandle` is represented as a raw pointer.
    pub is_challenge_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension, // `ConfigExtension*` is represented as a raw pointer.
}

impl Default for PowConfig {
    fn default() -> Self {
        PowConfig {
            stream: std::ptr::null_mut(),
            is_challenge_on_device: false,
            is_result_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

extern "C" {
    #[link_name = "pow"]
    pub fn pow_ffi(
        hasher: HasherHandle,
        challenge: *const u8,
        challenge_size: u32,
        padding_size: u32,
        bits: u8,
        config: *const PowConfig,
        found: *mut bool,
        nonce: *mut u64,
        mined_hash: *mut u64,
    ) -> eIcicleError;
}

pub fn pow_solver(
    hasher: &Hasher,
    challenge: &(impl HostOrDeviceSlice<u8> + ?Sized),
    bits: u8,
    config: &PowConfig,
    found: &mut (impl HostOrDeviceSlice<bool> + ?Sized),
    nonce: &mut (impl HostOrDeviceSlice<u64> + ?Sized),
    mined_hash: &mut (impl HostOrDeviceSlice<u64> + ?Sized),
) -> eIcicleError {
    if !(found.is_on_device() == nonce.is_on_device() && nonce.is_on_device() == mined_hash.is_on_device()) {
        panic!("found, nonce and mined_hash must be allocated on the same device");
    }
    if challenge.is_on_device() && !challenge.is_on_active_device() {
        panic!("challenge is allocated on an inactive device");
    }
    if found.is_on_device() && !found.is_on_active_device() {
        panic!("found is allocated on an inactive device");
    }
    if nonce.is_on_device() && !nonce.is_on_active_device() {
        panic!("nonce is allocated on an inactive device");
    }
    if mined_hash.is_on_device() && !mined_hash.is_on_active_device() {
        panic!("mined_hash is allocated on an inactive device");
    }
    let mut cfg = config.clone();
    cfg.is_challenge_on_device = challenge.is_on_device();
    cfg.is_result_on_device = found.is_on_device();

    let result = unsafe {
        pow_ffi(
            hasher.handle,
            challenge.as_ptr(),
            challenge.len() as u32,
            PADDING_SIZE,
            bits,
            &cfg as *const PowConfig,
            found.as_mut_ptr() as *mut bool,
            nonce.as_mut_ptr() as *mut u64,
            mined_hash.as_mut_ptr() as *mut u64,
        )
    };
    result
}