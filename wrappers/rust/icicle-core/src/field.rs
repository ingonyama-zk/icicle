#[macro_export]
macro_rules! impl_field {
    (
        $field_name:literal,
        $field_mod_extern:ident,
        $field_type:ident,
        $num_limbs:ident
    ) => {
        mod $field_mod_extern {
            use super::$field_type;
            use std::ffi::c_void;

            extern "C" {
                #[link_name = concat!($field_name, "_from_u32")]
                pub(crate) fn from_u32(val: u32, res: *mut $field_type) -> c_void;

                #[link_name = concat!($field_name, "_to_montgomery")]
                pub(crate) fn to_mont(val: *const $field_type, res: *mut $field_type) -> c_void;

                #[link_name = concat!($field_name, "_from_montgomery")]
                pub(crate) fn from_mont(val: *const $field_type, res: *mut $field_type) -> c_void;
                
                #[link_name = concat!($field_name, "_add")]
                pub(crate) fn add(a: *const $field_type, b: *const $field_type, result: *mut $field_type) -> c_void;

                #[link_name = concat!($field_name, "_sub")]
                pub(crate) fn sub(a: *const $field_type, b: *const $field_type, result: *mut $field_type) -> c_void;

                #[link_name = concat!($field_name, "_mul")]
                pub(crate) fn mul(a: *const $field_type, b: *const $field_type, result: *mut $field_type) -> c_void;

                #[link_name = concat!($field_name, "_inv")]
                pub(crate) fn inv(a: *const $field_type, result: *mut $field_type) -> c_void;
            }
        }

        #[derive(PartialEq, Copy, Clone, Default)]
        #[repr(C)]
        pub struct $field_type {
            limbs: [u32; $num_limbs],
        }
    
        unsafe impl Send for $field_type {}
        unsafe impl Sync for $field_type {}
    
        // TODO - convert from montgomery
        impl Display for $field_type {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "0x")?;
                for &b in self
                    .limbs
                    .iter()
                    .rev()
                {
                    write!(f, "{:08x}", b)?;
                }
                Ok(())
            }
        }
    
        // TODO - convert from montgomery
        impl Debug for $field_type {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{}", self.to_string())
            }
        }
    
        impl Into<[u32; $num_limbs]> for $field_type {
            fn into(self) -> [u32; $num_limbs] {
                let mut res = self;
                unsafe { $field_mod_extern::to_mont(&res, &mut res); }
                res.limbs
            }
        }
    
        impl From<[u32; $num_limbs]> for $field_type {
            fn from(limbs: [u32; $num_limbs]) -> Self {
                let mut res = Self { limbs };
                unsafe { $field_mod_extern::to_mont(&res, &mut res); }
                res
            }
        }

        impl Mul for $field_type {
            type Output = Self;
            fn mul(self, second: Self) -> Self {
                let mut result = $field_type::zero();
                unsafe {
                    $field_mod_extern::mul(
                        &self as *const $field_type,
                        &second as *const $field_type,
                        &mut result as *mut $field_type,
                    );
                }
                result
            }
        }

        impl Add for $field_type {
            type Output = Self;
            fn add(self, second: Self) -> Self {
                let mut result = $field_type::zero();
                unsafe {
                    $field_mod_extern::add(
                        &self as *const $field_type,
                        &second as *const $field_type,
                        &mut result as *mut $field_type,
                    );
                }
                result
            }
        }

        impl Sub for $field_type {
            type Output = Self;
            fn sub(self, second: Self) -> Self {
                let mut result = $field_type::zero();
                unsafe {
                    $field_mod_extern::sub(
                        &self as *const $field_type,
                        &second as *const $field_type,
                        &mut result as *mut $field_type,
                    );
                }
                result
            }
        }
    
        impl FieldImpl for $field_type {
            type Repr = [u32; $num_limbs];
        
            fn to_bytes_le(&self) -> Vec<u8> {
                self.limbs
                    .iter()
                    .map(|limb| {
                        limb.to_le_bytes()
                            .to_vec()
                    })
                    .flatten()
                    .collect::<Vec<_>>()
            }
        
            // please note that this function zero-pads if there are not enough bytes
            // and only takes the first bytes in there are too many of them
            fn from_bytes_le(bytes: &[u8]) -> Self {
                let mut limbs: [u32; $num_limbs] = [0; $num_limbs];
                for (i, chunk) in bytes
                    .chunks(4)
                    .take($num_limbs)
                    .enumerate()
                {
                    let mut chunk_array: [u8; 4] = [0; 4];
                    chunk_array[..chunk.len()].clone_from_slice(chunk);
                    limbs[i] = u32::from_le_bytes(chunk_array);
                }
                Self::from(limbs)
            }
        
            fn from_hex(s: &str) -> Self {
                let mut bytes = Vec::from_hex(if s.starts_with("0x") { &s[2..] } else { s }).expect("Invalid hex string");
                bytes.reverse();
                Self::from_bytes_le(&bytes)
            }
        
            fn zero() -> Self {
                Self::default()
            }
        
            fn one() -> Self {
                FieldImpl::from_u32(1)
            }
        
            fn from_u32(val: u32) -> Self {
                let mut res = Self::zero();
                unsafe { $field_mod_extern::from_u32(val, &mut res); }
                res
            }

            fn sqr(self) -> Self {
                self * self
            }

            fn inv(self) -> Self {
                let mut result = $field_type::zero();
                unsafe {
                    $field_mod_extern::inv(&self as *const $field_type, &mut result as *mut $field_type);
                }
                result
            }
        }
    };
}

#[macro_export]
macro_rules! impl_scalar_field {
    (
        $field_name:literal,
        $field_mod_extern:ident,
        $scalar_field_mod_extern:ident,
        $field_type:ident,
        $num_limbs:ident
    ) => {
        // TODO - make sure curves base fields use the same base_field specifier as in cpp for $field_mod_extern
        impl_field!($field_name, $field_mod_extern, $field_type, $num_limbs);

        mod $scalar_field_mod_extern {
            use super::$field_type;
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($field_name, "_generate_scalars")]
                pub(crate) fn _generate_scalars(scalars: *mut $field_type, size: usize);
        
                #[link_name = concat!($field_name, "_scalar_convert_montgomery")]
                pub(crate) fn _convert_scalars_montgomery(
                    scalars: *const $field_type,
                    size: u64,
                    is_into: bool,
                    config: &VecOpsConfig,
                    output: *mut $field_type,
                ) -> eIcicleError;
            }
        }

        impl ScalarImpl for $field_type {}

        impl GenerateRandom for $field_type {
            fn generate_random(size: usize) -> Vec<$field_type> {
                let mut res = vec![$field_type::zero(); size];
                unsafe { $scalar_field_mod_extern::_generate_scalars(&mut res[..] as *mut _ as *mut $field_type, size) };
                res
            }
        }
    
        impl MontgomeryConvertible for $field_type {
            fn to_mont(
                values: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                stream: &IcicleStream,
            ) -> eIcicleError {
                // check device slice is on active device
                if values.is_on_device() && !values.is_on_active_device() {
                    panic!("input not allocated on the active device");
                }
                let mut config = VecOpsConfig::default();
                config.is_a_on_device = values.is_on_device();
                config.is_async = !stream.is_null();
                config.stream_handle = (&*stream).into();
                unsafe {
                    let scalars_ptr = values.as_mut_ptr();
                    $scalar_field_mod_extern::_convert_scalars_montgomery(scalars_ptr, values.len() as u64, true, &config, scalars_ptr )
                }
            }
    
            fn from_mont(
                values: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                stream: &IcicleStream,
            ) -> eIcicleError {
                // check device slice is on active device
                if values.is_on_device() && !values.is_on_active_device() {
                    panic!("input not allocated on the active device");
                }
                let mut config = VecOpsConfig::default();
                config.is_a_on_device = values.is_on_device();
                config.is_async = !stream.is_null();
                config.stream_handle = (&*stream).into();
                unsafe {
                    let scalars_ptr = values.as_mut_ptr();
                    $scalar_field_mod_extern::_convert_scalars_montgomery(scalars_ptr, values.len() as u64, false, &config, scalars_ptr )
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_field_tests {
    (
        $field_type:ident
    ) => {
        pub mod test_field {
            use super::*;
            use icicle_runtime::test_utilities;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_field_convert_montgomery() {
                initialize();
                check_field_convert_montgomery::<$field_type>()
            }

            #[test]
            fn test_field_equality() {
                initialize();
                check_field_equality::<$field_type>()
            }
 
            #[test]
            fn test_field_arithmetic() {
                check_field_arithmetic::<$field_type>()
            }
        }
    };
}
