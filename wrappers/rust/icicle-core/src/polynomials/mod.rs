#[macro_export]
macro_rules! impl_polynomial_api {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident
    ) => {
        mod $field_prefix_ident {
            use crate::polynomials::*;
            use icicle_core::traits::FieldImpl;
            use icicle_cuda_runtime::memory::HostOrDeviceSlice;
            use std::cmp;
            use std::ffi::c_void;
            use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub};

            type PolynomialHandle = *const c_void;

            extern "C" {
                #[link_name = concat!($field_prefix, "init_cuda_backend")]
                fn init_cuda_backend() -> bool;

                #[link_name = concat!($field_prefix, "polynomial_create_from_coefficients")]
                fn create_from_coeffs(coeffs: *const $field, size: usize) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_create_from_rou_evaluations")]
                fn create_from_rou_evals(coeffs: *const $field, size: usize) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_clone")]
                fn clone(p: PolynomialHandle) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_delete")]
                fn delete(ptr: PolynomialHandle);

                #[link_name = concat!($field_prefix, "polynomial_print")]
                fn print(ptr: PolynomialHandle);

                #[link_name = concat!($field_prefix, "polynomial_add")]
                fn add(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_add_inplace")]
                fn add_inplace(a: PolynomialHandle, b: PolynomialHandle) -> c_void;

                #[link_name = concat!($field_prefix, "polynomial_subtract")]
                fn subtract(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_multiply")]
                fn multiply(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_multiply_by_scalar")]
                fn multiply_by_scalar(a: PolynomialHandle, b: &$field) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_quotient")]
                fn quotient(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_remainder")]
                fn remainder(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_division")]
                fn divide(a: PolynomialHandle, b: PolynomialHandle, q: *mut PolynomialHandle, r: *mut PolynomialHandle);

                #[link_name = concat!($field_prefix, "polynomial_divide_by_vanishing")]
                fn div_by_vanishing(a: PolynomialHandle, deg: u64) -> PolynomialHandle;

                #[link_name = concat!($field_prefix, "polynomial_add_monomial_inplace")]
                fn add_monomial_inplace(a: PolynomialHandle, monomial_coeff: &$field, monomial: u64) -> c_void;

                #[link_name = concat!($field_prefix, "polynomial_sub_monomial_inplace")]
                fn sub_monomial_inplace(a: PolynomialHandle, monomial_coeff: &$field, monomial: u64) -> c_void;

                #[link_name = concat!($field_prefix, "polynomial_evaluate")]
                fn eval(a: PolynomialHandle, x: &$field) -> $field;

                #[link_name = concat!($field_prefix, "polynomial_evaluate_on_domain")]
                fn eval_on_domain(a: PolynomialHandle, domain: *const $field, domain_size: u64, evals: *mut $field);

                #[link_name = concat!($field_prefix, "polynomial_degree")]
                fn degree(a: PolynomialHandle) -> i64;

                #[link_name = concat!($field_prefix, "polynomial_get_coeff")]
                fn get_coeff(a: PolynomialHandle, idx: u64) -> $field;

                #[link_name = concat!($field_prefix, "polynomial_copy_coeffs_range")]
                fn copy_coeffs(a: PolynomialHandle, host_coeffs: *mut $field, start_idx: i64, end_idx: i64) -> i64;

            }

            pub struct Polynomial {
                handle: PolynomialHandle,
            }

            impl Polynomial {
                pub fn init_cuda_backend() -> bool {
                    unsafe { init_cuda_backend() }
                }

                pub fn from_coeffs<S: HostOrDeviceSlice<$field> + ?Sized>(coeffs: &S, size: usize) -> Self {
                    unsafe {
                        Polynomial {
                            handle: create_from_coeffs(coeffs.as_ptr(), size),
                        }
                    }
                }

                pub fn from_rou_evals<S: HostOrDeviceSlice<$field> + ?Sized>(evals: &S, size: usize) -> Self {
                    unsafe {
                        Polynomial {
                            handle: create_from_rou_evals(evals.as_ptr(), size),
                        }
                    }
                }
                pub fn clone(&self) -> Polynomial {
                    unsafe {
                        Polynomial {
                            handle: clone(self.handle),
                        }
                    }
                }

                pub fn divide(&self, denominator: &Polynomial) -> (Polynomial, Polynomial) {
                    let mut q_handle: PolynomialHandle = std::ptr::null_mut();
                    let mut r_handle: PolynomialHandle = std::ptr::null_mut();
                    unsafe {
                        divide(self.handle, denominator.handle, &mut q_handle, &mut r_handle);
                    }
                    (Polynomial { handle: q_handle }, Polynomial { handle: r_handle })
                }

                pub fn div_by_vanishing(&self, degree: u64) -> Polynomial {
                    unsafe {
                        Polynomial {
                            handle: div_by_vanishing(self.handle, degree),
                        }
                    }
                }

                pub fn add_monomial_inplace(&mut self, monomial_coeff: &$field, monomial: u64) {
                    unsafe {
                        add_monomial_inplace(self.handle, monomial_coeff, monomial);
                    }
                }

                pub fn sub_monomial_inplace(&mut self, monomial_coeff: &$field, monomial: u64) {
                    unsafe {
                        sub_monomial_inplace(self.handle, monomial_coeff, monomial);
                    }
                }

                pub fn eval(&self, x: &$field) -> $field {
                    unsafe { eval(self.handle, x) }
                }

                pub fn eval_on_domain<S: HostOrDeviceSlice<$field> + ?Sized>(&self, domain: &S, evals: &mut S) {
                    assert!(
                        domain.len() <= evals.len(),
                        "eval_on_domain(): eval size must not be smaller then domain"
                    );
                    unsafe {
                        eval_on_domain(
                            self.handle,
                            domain.as_ptr(),
                            domain.len() as u64,
                            evals.as_mut_ptr(),
                        );
                    }
                }

                pub fn get_nof_coeffs(&self) -> u64 {
                    unsafe {
                        // returns total #coeffs. Not copying when null
                        let nof_coeffs = copy_coeffs(self.handle, std::ptr::null_mut(), 0, 0);
                        nof_coeffs as u64
                    }
                }

                pub fn get_coeff(&self, idx: u64) -> $field {
                    unsafe { get_coeff(self.handle, idx) }
                }

                pub fn copy_coeffs<S: HostOrDeviceSlice<$field> + ?Sized>(&self, start_idx: u64, coeffs: &mut S) {
                    let coeffs_len = coeffs.len() as u64;
                    let nof_coeffs = self.get_nof_coeffs();
                    let end_idx = cmp::min(nof_coeffs, start_idx + coeffs_len - 1);

                    unsafe {
                        copy_coeffs(self.handle, coeffs.as_mut_ptr(), start_idx as i64, end_idx as i64);
                    }
                }

                pub fn degree(&self) -> i64 {
                    unsafe { degree(self.handle) }
                }

                // TODO Yuval: implement Display trait
                pub fn print(&self) {
                    unsafe {
                        print(self.handle);
                    }
                }
            }

            impl Drop for Polynomial {
                fn drop(&mut self) {
                    unsafe {
                        delete(self.handle);
                    }
                }
            }

            impl Add for &Polynomial {
                type Output = Polynomial;

                fn add(self: Self, rhs: Self) -> Self::Output {
                    unsafe {
                        Polynomial {
                            handle: add(self.handle, rhs.handle),
                        }
                    }
                }
            }

            impl AddAssign<&Polynomial> for Polynomial {
                fn add_assign(&mut self, other: &Polynomial) {
                    unsafe { add_inplace(self.handle, other.handle) };
                }
            }

            impl Sub for &Polynomial {
                type Output = Polynomial;

                fn sub(self: Self, rhs: Self) -> Self::Output {
                    unsafe {
                        Polynomial {
                            handle: subtract(self.handle, rhs.handle),
                        }
                    }
                }
            }

            impl Mul for &Polynomial {
                type Output = Polynomial;

                fn mul(self: Self, rhs: Self) -> Self::Output {
                    unsafe {
                        Polynomial {
                            handle: multiply(self.handle, rhs.handle),
                        }
                    }
                }
            }

            // poly * scalar
            impl Mul<&$field> for &Polynomial {
                type Output = Polynomial;

                fn mul(self: Self, rhs: &$field) -> Self::Output {
                    unsafe {
                        Polynomial {
                            handle: multiply_by_scalar(self.handle, rhs),
                        }
                    }
                }
            }
            // scalar * poly
            impl Mul<&Polynomial> for &$field {
                type Output = Polynomial;

                fn mul(self, rhs: &Polynomial) -> Self::Output {
                    unsafe {
                        Polynomial {
                            handle: multiply_by_scalar(rhs.handle, self),
                        }
                    }
                }
            }

            impl Div for &Polynomial {
                type Output = Polynomial;

                fn div(self: Self, rhs: Self) -> Self::Output {
                    unsafe {
                        Polynomial {
                            handle: quotient(self.handle, rhs.handle),
                        }
                    }
                }
            }

            impl Rem for &Polynomial {
                type Output = Polynomial;

                fn rem(self: Self, rhs: Self) -> Self::Output {
                    unsafe {
                        Polynomial {
                            handle: remainder(self.handle, rhs.handle),
                        }
                    }
                }
            }
        }
    };
}
