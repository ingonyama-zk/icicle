use std::fmt::{Debug, Display};

pub trait PrimeField:
    Display + Debug + PartialEq + Copy + Clone + Into<Self::Limbs> + From<Self::Limbs> + Send + Sync
{
    const LIMBS_SIZE: usize;
    type Limbs: AsRef<[u32]> + AsMut<[u32]>;

    fn limbs(&self) -> &Self::Limbs;
    fn limbs_mut(&mut self) -> &mut Self::Limbs;

    fn from_u32(val: u32) -> Self;

    fn zero() -> Self {
        Self::from_u32(0)
    }

    fn one() -> Self {
        Self::from_u32(1)
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.limbs()
            .as_ref()
            .iter()
            .flat_map(|&x| {
                x.to_le_bytes()
                    .to_vec()
            })
            .collect()
    }

    // please note that this function zero-pads if there are not enough bytes
    // and only takes the first bytes in there are too many of them
    fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut result = Self::zero();
        let limbs = result
            .limbs_mut()
            .as_mut();
        for (i, chunk) in bytes
            .chunks(4)
            .take(Self::LIMBS_SIZE)
            .enumerate()
        {
            let mut chunk_array: [u8; 4] = [0; 4];
            chunk_array[..chunk.len()].clone_from_slice(chunk);
            limbs[i] = u32::from_le_bytes(chunk_array);
        }
        result
    }

    fn from_hex(s: &str) -> Self {
        let s = if s.starts_with("0x") { &s[2..] } else { s };
        let mut bytes = hex::decode(s).expect("Invalid hex string");
        bytes.reverse();
        Self::from_bytes_le(&bytes)
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "0x")?;
        for &b in self
            .limbs()
            .as_ref()
            .iter()
            .rev()
        {
            write!(f, "{:08x}", b)?;
        }
        Ok(())
    }

    fn debug_fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[macro_export]
macro_rules! impl_field {
    (
        $field:ident,
        $field_prefix:literal,
        $num_limbs:ident,
        $use_ffi:expr
    ) => {
        #[derive(Copy, Clone)]
        #[repr(C)]
        pub struct $field {
            limbs: [u32; $num_limbs],
        }

        impl Default for $field {
            fn default() -> Self {
                Self::zero()
            }
        }

        impl PartialEq for $field {
            fn eq(&self, other: &Self) -> bool {
                if $use_ffi {
                    extern "C" {
                        #[link_name = concat!($field_prefix, "_eq")]
                        pub(crate) fn _eq(left: *const $field, right: *const $field, result: *mut bool);
                    }
                    let mut result = false;
                    unsafe {
                        _eq(
                            self.limbs
                                .as_ptr() as *const $field,
                            other
                                .limbs
                                .as_ptr() as *const $field,
                            &mut result,
                        );
                    }
                    result
                } else {
                    self.limbs == other.limbs
                }
            }
        }

        impl Into<[u32; $num_limbs]> for $field {
            fn into(self) -> [u32; $num_limbs] {
                self.limbs
            }
        }

        impl From<[u32; $num_limbs]> for $field {
            fn from(limbs: [u32; $num_limbs]) -> Self {
                Self { limbs }
            }
        }

        impl PrimeField for $field {
            const LIMBS_SIZE: usize = $num_limbs;
            type Limbs = [u32; $num_limbs];

            fn limbs(&self) -> &Self::Limbs {
                &self.limbs
            }

            fn limbs_mut(&mut self) -> &mut Self::Limbs {
                &mut self.limbs
            }

            fn from_u32(val: u32) -> Self {
                if $use_ffi {
                    extern "C" {
                        #[link_name = concat!($field_prefix, "_from_u32")]
                        pub(crate) fn from_u32(val: u32, result: *mut $field);
                    }

                    let mut limbs = [0u32; $num_limbs];

                    unsafe {
                        // Convert `val` into field representation using an external FFI call.
                        // Casting `limbs` ensures compatibility without tightly coupling `FieldConfig` and `Field`.
                        from_u32(val, limbs.as_mut_ptr() as *mut $field);
                    }

                    Self { limbs }
                } else {
                    let mut limbs = [0u32; $num_limbs];
                    limbs[0] = val;
                    Self { limbs }
                }
            }
        }

        impl Display for $field {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                <Self as PrimeField>::fmt(self, f)
            }
        }

        impl Debug for $field {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                <Self as PrimeField>::debug_fmt(self, f)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_field_arithmetic {
    (
        $field:ident,
        $field_prefix:literal,
        $field_prefix_ident:ident
    ) => {
        mod $field_prefix_ident {
            use super::$field;

            extern "C" {
                #[link_name = concat!($field_prefix, "_add")]
                pub(crate) fn add(a: *const $field, b: *const $field, result: *mut $field);

                #[link_name = concat!($field_prefix, "_sub")]
                pub(crate) fn sub(a: *const $field, b: *const $field, result: *mut $field);

                #[link_name = concat!($field_prefix, "_mul")]
                pub(crate) fn mul(a: *const $field, b: *const $field, result: *mut $field);

                #[link_name = concat!($field_prefix, "_sqr")]
                pub(crate) fn sqr(a: *const $field, result: *mut $field);

                #[link_name = concat!($field_prefix, "_inv")]
                pub(crate) fn inv(a: *const $field, result: *mut $field);

                #[link_name = concat!($field_prefix, "_pow")]
                pub(crate) fn pow(a: *const $field, exp: usize, result: *mut $field);
            }
        }

        impl Add for $field {
            type Output = Self;

            fn add(self, second: Self) -> Self {
                let mut result = Self::zero();
                unsafe {
                    $field_prefix_ident::add(
                        &self as *const $field,
                        &second as *const $field,
                        &mut result as *mut $field,
                    );
                }
                result
            }
        }

        impl Sub for $field {
            type Output = Self;

            fn sub(self, second: Self) -> Self {
                let mut result = Self::zero();
                unsafe {
                    $field_prefix_ident::sub(
                        &self as *const $field,
                        &second as *const $field,
                        &mut result as *mut $field,
                    );
                }
                result
            }
        }

        impl Mul for $field {
            type Output = Self;

            fn mul(self, second: Self) -> Self {
                let mut result = Self::zero();
                unsafe {
                    $field_prefix_ident::mul(
                        &self as *const $field,
                        &second as *const $field,
                        &mut result as *mut $field,
                    );
                }
                result
            }
        }

        impl Arithmetic for $field {
            fn sqr(&self) -> Self {
                let mut result = Self::zero();
                unsafe {
                    $field_prefix_ident::sqr(self as *const $field, &mut result as *mut $field);
                }
                result
            }

            fn inv(&self) -> Self {
                let mut result = Self::zero();
                unsafe {
                    $field_prefix_ident::inv(self as *const $field, &mut result as *mut $field);
                }
                result
            }

            fn pow(&self, exp: usize) -> Self {
                let mut result = Self::zero();
                unsafe {
                    $field_prefix_ident::pow(self as *const $field, exp, &mut result as *mut $field);
                }
                result
            }
        }
    };
}

#[macro_export]
macro_rules! impl_montgomery_convertible {
    (
        $field:ident,
        $convert_montgomery_function_name:ident
    ) => {
        impl $field {
            fn convert_montgomery(
                values: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                stream: &IcicleStream,
                is_into: bool,
            ) -> eIcicleError {
                extern "C" {
                    fn $convert_montgomery_function_name(
                        values: *const $field,
                        size: u64,
                        is_into: bool,
                        config: &VecOpsConfig,
                        output: *mut $field,
                    ) -> eIcicleError;
                }

                // check device slice is on active device
                if values.is_on_device() && !values.is_on_active_device() {
                    panic!("input not allocated on the active device");
                }
                let mut config = VecOpsConfig::default();
                config.is_a_on_device = values.is_on_device();
                config.is_async = !stream.is_null();
                config.stream_handle = (&*stream).into();
                unsafe {
                    $convert_montgomery_function_name(
                        values.as_ptr(),
                        values.len() as u64,
                        is_into,
                        &config,
                        values.as_mut_ptr(),
                    )
                }
            }
        }

        impl MontgomeryConvertible for $field {
            fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
                $field::convert_montgomery(values, stream, true)
            }

            fn from_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
                $field::convert_montgomery(values, stream, false)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_generate_random {
    (
        $field:ident,
        $generate_random_function_name:ident
    ) => {
        impl GenerateRandom for $field {
            fn generate_random(size: usize) -> Vec<$field> {
                extern "C" {
                    pub(crate) fn $generate_random_function_name(scalars: *mut $field, size: usize);
                }

                let mut res = vec![$field::zero(); size];
                unsafe { $generate_random_function_name(&mut res[..] as *mut _ as *mut $field, size) };
                res
            }
        }
    };
}

#[macro_export]
macro_rules! impl_field_tests {
    (
        $field:ident
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
                check_field_convert_montgomery::<$field>()
            }

            #[test]
            fn test_field_arithmetic() {
                check_field_arithmetic::<$field>()
            }
        }
    };
}
