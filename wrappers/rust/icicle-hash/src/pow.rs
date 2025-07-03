use icicle_core::hash::{Hasher, HasherHandle};
use icicle_runtime::{
    config::ConfigExtension, eIcicleError, memory::HostOrDeviceSlice, IcicleError, IcicleStreamHandle,
};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PowConfig {
    pub stream: IcicleStreamHandle, // `icicleStreamHandle` is represented as a raw pointer.
    is_challenge_on_device: bool,
    pub padding_size: u32,
    pub is_async: bool,
    pub ext: ConfigExtension, // `ConfigExtension*` is represented as a raw pointer.
}

impl Default for PowConfig {
    fn default() -> Self {
        PowConfig {
            stream: std::ptr::null_mut(),
            is_challenge_on_device: false,
            padding_size: 24,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

extern "C" {
    #[link_name = "proof_of_work"]
    pub fn pow_ffi(
        hasher: HasherHandle,
        challenge: *const u8,
        challenge_size: u32,
        solution_bits: u8,
        config: *const PowConfig,
        found: *mut bool,
        nonce: *mut u64,
        mined_hash: *mut u64,
    ) -> eIcicleError;
}
extern "C" {
    #[link_name = "proof_of_work_verify"]
    pub fn pow_verify_ffi(
        hasher: HasherHandle,
        challenge: *const u8,
        challenge_size: u32,
        solution_bits: u8,
        config: *const PowConfig,
        nonce: u64,
        is_correct: *mut bool,
        mined_hash: *mut u64,
    ) -> eIcicleError;
}

pub fn pow_solver(
    hasher: &Hasher,
    challenge: &(impl HostOrDeviceSlice<u8> + ?Sized),
    solution_bits: u8,
    config: &PowConfig,
    found: &mut bool,
    nonce: &mut u64,
    mined_hash: &mut u64,
) -> Result<(), IcicleError> {
    if !(1..=60).contains(&solution_bits) {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            "invalid solution_bits value",
        ));
    }
    if challenge.is_on_device() && !challenge.is_on_active_device() {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            "challenge is allocated on an inactive device",
        ));
    }
    let mut cfg = config.clone();
    cfg.is_challenge_on_device = challenge.is_on_device();

    unsafe {
        pow_ffi(
            hasher.handle,
            challenge.as_ptr(),
            challenge.len() as u32,
            solution_bits,
            &cfg as *const PowConfig,
            found as *mut bool,
            nonce as *mut u64,
            mined_hash as *mut u64,
        )
        .wrap()
    }
}

pub fn pow_verify(
    hasher: &Hasher,
    challenge: &(impl HostOrDeviceSlice<u8> + ?Sized),
    solution_bits: u8,
    config: &PowConfig,
    nonce: u64,
    is_correct: &mut bool,
    mined_hash: &mut u64,
) -> Result<(), IcicleError> {
    if !(1..=60).contains(&solution_bits) {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            "invalid solution_bits value",
        ));
    }
    if challenge.is_on_device() && !challenge.is_on_active_device() {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            "challenge is allocated on an inactive device",
        ));
    }
    let mut cfg = config.clone();
    cfg.is_challenge_on_device = challenge.is_on_device();

    unsafe {
        pow_verify_ffi(
            hasher.handle,
            challenge.as_ptr(),
            challenge.len() as u32,
            solution_bits,
            &cfg as *const PowConfig,
            nonce,
            is_correct as *mut bool,
            mined_hash as *mut u64,
        )
        .wrap()
    }
}
