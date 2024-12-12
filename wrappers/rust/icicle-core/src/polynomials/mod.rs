use crate::traits::FieldImpl;
use icicle_runtime::memory::HostOrDeviceSlice;

pub trait UnivariatePolynomial
where
    Self::Field: FieldImpl,
{
    type Field;

    fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, size: usize) -> Self;
    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, size: usize) -> Self;
    fn divide(&self, denominator: &Self) -> (Self, Self)
    where
        Self: Sized;
    fn div_by_vanishing(&self, degree: u64) -> Self;
    fn add_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);
    fn sub_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);
    fn slice(&self, offset: u64, stride: u64, size: u64) -> Self;
    fn even(&self) -> Self;
    fn odd(&self) -> Self;
    fn eval(&self, x: &Self::Field) -> Self::Field;
    fn degree(&self) -> i64;
    fn eval_on_domain<D: HostOrDeviceSlice<Self::Field> + ?Sized, E: HostOrDeviceSlice<Self::Field> + ?Sized>(
        &self,
        domain: &D,
        evals: &mut E,
    );
    fn eval_on_rou_domain<E: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, domain_log_size: u64, evals: &mut E);
    fn get_nof_coeffs(&self) -> u64;
    fn get_coeff(&self, idx: u64) -> Self::Field;
    fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, start_idx: u64, coeffs: &mut S);
}

#[macro_export]
macro_rules! impl_univariate_polynomial_api {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident
    ) => {
        use icicle_core::{polynomials::UnivariatePolynomial, traits::FieldImpl};
        use icicle_runtime::memory::{DeviceSlice, HostOrDeviceSlice};
        use std::{
            clone, cmp,
            ffi::c_void,
            ops::{Add, AddAssign, Div, Mul, Rem, Sub},
            ptr, slice,
        };

        type PolynomialHandle = *const c_void;

        extern "C" {
            #[link_name = concat!($field_prefix, "_polynomial_create_from_coefficients")]
            fn create_from_coeffs(coeffs: *const $field, size: usize) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_create_from_rou_evaluations")]
            fn create_from_rou_evals(coeffs: *const $field, size: usize) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_clone")]
            fn clone(p: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_delete")]
            fn delete(ptr: PolynomialHandle);

            #[link_name = concat!($field_prefix, "_polynomial_print")]
            fn print(ptr: PolynomialHandle);

            #[link_name = concat!($field_prefix, "_polynomial_add")]
            fn add(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_add_inplace")]
            fn add_inplace(a: PolynomialHandle, b: PolynomialHandle) -> c_void;

            #[link_name = concat!($field_prefix, "_polynomial_subtract")]
            fn subtract(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_multiply")]
            fn multiply(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_multiply_by_scalar")]
            fn multiply_by_scalar(a: PolynomialHandle, b: &$field) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_quotient")]
            fn quotient(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_remainder")]
            fn remainder(a: PolynomialHandle, b: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_division")]
            fn divide(a: PolynomialHandle, b: PolynomialHandle, q: *mut PolynomialHandle, r: *mut PolynomialHandle);

            #[link_name = concat!($field_prefix, "_polynomial_divide_by_vanishing")]
            fn div_by_vanishing(a: PolynomialHandle, deg: u64) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_add_monomial_inplace")]
            fn add_monomial_inplace(a: PolynomialHandle, monomial_coeff: &$field, monomial: u64) -> c_void;

            #[link_name = concat!($field_prefix, "_polynomial_sub_monomial_inplace")]
            fn sub_monomial_inplace(a: PolynomialHandle, monomial_coeff: &$field, monomial: u64) -> c_void;

            #[link_name = concat!($field_prefix, "_polynomial_slice")]
            fn slice(a: PolynomialHandle, offset: u64, stride: u64, size: u64) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_even")]
            fn even(a: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_odd")]
            fn odd(a: PolynomialHandle) -> PolynomialHandle;

            #[link_name = concat!($field_prefix, "_polynomial_evaluate_on_domain")]
            fn eval_on_domain(a: PolynomialHandle, domain: *const $field, domain_size: u64, evals: *mut $field);

            #[link_name = concat!($field_prefix, "_polynomial_evaluate_on_rou_domain")]
            fn eval_on_rou_domain(a: PolynomialHandle, domain_log_size: u64, evals: *mut $field);

            #[link_name = concat!($field_prefix, "_polynomial_degree")]
            fn degree(a: PolynomialHandle) -> i64;

            #[link_name = concat!($field_prefix, "_polynomial_copy_coeffs_range")]
            fn copy_coeffs(a: PolynomialHandle, host_coeffs: *mut $field, start_idx: u64, end_idx: u64) -> u64;

            #[link_name = concat!($field_prefix, "_polynomial_get_coeffs_raw_ptr")]
            fn get_coeffs_ptr(a: PolynomialHandle, len: *mut u64) -> *mut $field;
        }

        pub struct DensePolynomial {
            handle: PolynomialHandle,
        }

        impl DensePolynomial {
            // TODO Yuval: implement Display trait
            pub fn print(&self) {
                unsafe {
                    print(self.handle);
                }
            }

            pub fn coeffs_mut_slice(&mut self) -> &mut DeviceSlice<$field> {
                unsafe {
                    let mut len: u64 = 0;
                    let mut coeffs_mut = get_coeffs_ptr(self.handle, &mut len);
                    let s = slice::from_raw_parts_mut(coeffs_mut, len as usize);
                    DeviceSlice::from_mut_slice(s)
                }
            }
        }

        impl UnivariatePolynomial for DensePolynomial {
            type Field = $field;

            fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, size: usize) -> Self {
                unsafe {
                    DensePolynomial {
                        handle: create_from_coeffs(coeffs.as_ptr(), size),
                    }
                }
            }

            fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, size: usize) -> Self {
                unsafe {
                    Self {
                        handle: create_from_rou_evals(evals.as_ptr(), size),
                    }
                }
            }

            fn divide(&self, denominator: &Self) -> (Self, Self) {
                let mut q_handle: PolynomialHandle = std::ptr::null_mut();
                let mut r_handle: PolynomialHandle = std::ptr::null_mut();
                unsafe {
                    divide(self.handle, denominator.handle, &mut q_handle, &mut r_handle);
                }
                (Self { handle: q_handle }, Self { handle: r_handle })
            }

            fn div_by_vanishing(&self, degree: u64) -> Self {
                unsafe {
                    Self {
                        handle: div_by_vanishing(self.handle, degree),
                    }
                }
            }

            fn add_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64) {
                unsafe {
                    add_monomial_inplace(self.handle, monomial_coeff, monomial);
                }
            }

            fn sub_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64) {
                unsafe {
                    sub_monomial_inplace(self.handle, monomial_coeff, monomial);
                }
            }

            fn slice(&self, offset: u64, stride: u64, size: u64) -> Self {
                unsafe {
                    Self {
                        handle: slice(self.handle, offset, stride, size),
                    }
                }
            }

            fn even(&self) -> Self {
                unsafe {
                    Self {
                        handle: even(self.handle),
                    }
                }
            }

            fn odd(&self) -> Self {
                unsafe {
                    Self {
                        handle: odd(self.handle),
                    }
                }
            }

            fn eval(&self, x: &Self::Field) -> Self::Field {
                let mut eval = Self::Field::zero();
                unsafe {
                    eval_on_domain(self.handle, x, 1, &mut eval);
                }
                eval
            }

            fn eval_on_domain<
                D: HostOrDeviceSlice<Self::Field> + ?Sized,
                E: HostOrDeviceSlice<Self::Field> + ?Sized,
            >(
                &self,
                domain: &D,
                evals: &mut E,
            ) {
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

            fn eval_on_rou_domain<E: HostOrDeviceSlice<Self::Field> + ?Sized>(
                &self,
                domain_log_size: u64,
                evals: &mut E,
            ) {
                assert!(
                    evals.len() >= 1 << domain_log_size,
                    "eval_on_rou_domain(): eval size must not be smaller than domain"
                );
                unsafe {
                    eval_on_rou_domain(self.handle, domain_log_size, evals.as_mut_ptr());
                }
            }

            fn get_nof_coeffs(&self) -> u64 {
                unsafe {
                    // returns total #coeffs. Not copying when null
                    let nof_coeffs = copy_coeffs(self.handle, std::ptr::null_mut(), 0, 0);
                    nof_coeffs
                }
            }

            fn get_coeff(&self, idx: u64) -> Self::Field {
                let mut coeff: Self::Field = Self::Field::zero();
                unsafe { copy_coeffs(self.handle, &mut coeff, idx, idx) };
                coeff
            }

            fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, start_idx: u64, coeffs: &mut S) {
                let coeffs_len = coeffs.len() as u64;
                let nof_coeffs = self.get_nof_coeffs();
                let end_idx = cmp::min(nof_coeffs, start_idx + coeffs_len - 1);

                unsafe {
                    copy_coeffs(self.handle, coeffs.as_mut_ptr(), start_idx, end_idx);
                }
            }

            fn degree(&self) -> i64 {
                unsafe { degree(self.handle) }
            }
        }

        impl Drop for DensePolynomial {
            fn drop(&mut self) {
                unsafe {
                    delete(self.handle);
                }
            }
        }

        impl Clone for DensePolynomial {
            fn clone(&self) -> Self {
                unsafe {
                    DensePolynomial {
                        handle: clone(self.handle),
                    }
                }
            }
        }

        impl Add for &DensePolynomial {
            type Output = DensePolynomial;

            fn add(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: add(self.handle, rhs.handle),
                    }
                }
            }
        }

        impl AddAssign<&DensePolynomial> for DensePolynomial {
            fn add_assign(&mut self, other: &DensePolynomial) {
                unsafe { add_inplace(self.handle, other.handle) };
            }
        }

        impl Sub for &DensePolynomial {
            type Output = DensePolynomial;

            fn sub(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: subtract(self.handle, rhs.handle),
                    }
                }
            }
        }

        impl Mul for &DensePolynomial {
            type Output = DensePolynomial;

            fn mul(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: multiply(self.handle, rhs.handle),
                    }
                }
            }
        }

        // poly * scalar
        impl Mul<&$field> for &DensePolynomial {
            type Output = DensePolynomial;

            fn mul(self: Self, rhs: &$field) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: multiply_by_scalar(self.handle, rhs),
                    }
                }
            }
        }

        // scalar * poly
        impl Mul<&DensePolynomial> for &$field {
            type Output = DensePolynomial;

            fn mul(self, rhs: &DensePolynomial) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: multiply_by_scalar(rhs.handle, self),
                    }
                }
            }
        }

        impl Div for &DensePolynomial {
            type Output = DensePolynomial;

            fn div(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: quotient(self.handle, rhs.handle),
                    }
                }
            }
        }

        impl Rem for &DensePolynomial {
            type Output = DensePolynomial;

            fn rem(self: Self, rhs: Self) -> Self::Output {
                unsafe {
                    DensePolynomial {
                        handle: remainder(self.handle, rhs.handle),
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_polynomial_tests {
    (
        $field_prefix_ident:ident,
        $field:ident
    ) => {
        use super::*;
        use icicle_core::ntt::{
            get_root_of_unity, initialize_domain, release_domain, NTTDomain, NTTInitDomainConfig,
            CUDA_NTT_FAST_TWIDDLES_MODE,
        };
        use icicle_core::vec_ops::{add_scalars, mul_scalars, sub_scalars, VecOps, VecOpsConfig};
        use icicle_runtime::memory::{DeviceVec, HostSlice};
        use icicle_runtime::test_utilities;
        use std::sync::Once;

        use icicle_core::traits::{FieldImpl, GenerateRandom};

        type Poly = DensePolynomial;

        pub fn init_domain<F>(max_size: u64, fast_twiddles_mode: bool)
        where
            F: FieldImpl + NTTDomain<F>,
        {
            let config = NTTInitDomainConfig::default();
            config
                .ext
                .set_bool(CUDA_NTT_FAST_TWIDDLES_MODE, fast_twiddles_mode);
            let rou = get_root_of_unity::<F>(max_size);
            initialize_domain(rou, &config).unwrap();
        }

        fn randomize_coeffs<F>(size: usize) -> Vec<F>
        where
            F: FieldImpl + GenerateRandom,
        {
            F::generate_random(size)
        }

        fn rand() -> $field {
            randomize_coeffs::<$field>(1)[0]
        }

        // Note: implementing field arithmetic (+,-,*) for fields via vec_ops since they are not implemented on host
        fn add(a: &$field, b: &$field) -> $field {
            let a = [a.clone()];
            let b = [b.clone()];
            let mut result = [$field::zero()];

            let cfg = VecOpsConfig::default();
            add_scalars(
                HostSlice::from_slice(&a),
                HostSlice::from_slice(&b),
                HostSlice::from_mut_slice(&mut result),
                &cfg,
            )
            .unwrap();
            result[0]
        }

        fn sub(a: &$field, b: &$field) -> $field {
            let a = [a.clone()];
            let b = [b.clone()];
            let mut result = [$field::zero()];

            let cfg = VecOpsConfig::default();
            sub_scalars(
                HostSlice::from_slice(&a),
                HostSlice::from_slice(&b),
                HostSlice::from_mut_slice(&mut result),
                &cfg,
            )
            .unwrap();
            result[0]
        }

        fn mul(a: &$field, b: &$field) -> $field {
            let a = [a.clone()];
            let b = [b.clone()];
            let mut result = [$field::zero()];

            let cfg = VecOpsConfig::default();
            mul_scalars(
                HostSlice::from_slice(&a),
                HostSlice::from_slice(&b),
                HostSlice::from_mut_slice(&mut result),
                &cfg,
            )
            .unwrap();
            result[0]
        }

        fn randomize_poly(size: usize) -> Poly {
            let coeffs = randomize_coeffs::<$field>(size);
            Poly::from_coeffs(HostSlice::from_slice(&coeffs), size)
        }

        static INIT: Once = Once::new();
        pub fn setup() -> () {
            INIT.call_once(move || {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();

                let domain_max_size: u64 = 1 << 20;
                init_domain::<$field>(domain_max_size, false /*=fast twiddle */);
            });
            test_utilities::test_set_main_device();
        }

        // Note: tests are prefixed with 'phase3' since they conflict with NTT tests domain.
        //       The poly tests are executed via 'cargo test phase3' as an additional step

        #[test]
        fn phase3_test_poly_eval() {
            setup();

            // testing correct evaluation of f(8) for f(x)=4x^2+2x+5
            let coeffs = [$field::from_u32(5), $field::from_u32(2), $field::from_u32(4)];
            let f = Poly::from_coeffs(HostSlice::from_slice(&coeffs), coeffs.len());
            let x = $field::from_u32(8);
            let f_x = f.eval(&x);
            assert_eq!(f_x, $field::from_u32(277));
        }

        #[test]
        fn phase3_test_poly_clone() {
            setup();

            // testing that the clone g(x) is independent of f(x) and cloned correctly
            let mut f = randomize_poly(8);
            let x = rand();
            let fx = f.eval(&x);

            let g = f.clone();
            f += &g;

            let gx = g.eval(&x);
            let new_fx = f.eval(&x);

            assert_eq!(fx, gx); // cloned correctly
            assert_eq!(add(&fx, &gx), new_fx);
        }

        #[test]
        fn phase3_test_poly_add_sub_mul() {
            setup();

            // testing add/sub operations
            let size = 1 << 10;
            let mut f = randomize_poly(size);
            let mut g = randomize_poly(size);

            let x = rand();
            let fx = f.eval(&x);
            let gx = g.eval(&x);

            let poly_add = &f + &g;
            let poly_sub = &f - &g;
            let poly_mul = &f * &g;

            assert_eq!(poly_add.eval(&x), add(&fx, &gx));
            assert_eq!(poly_sub.eval(&x), sub(&fx, &gx));
            assert_eq!(poly_mul.eval(&x), mul(&fx, &gx));

            // test scalar multiplication
            let s1 = rand();
            let s2 = rand();
            let poly_mul_s1 = &f * &s1;
            let poly_mul_s2 = &s2 * &f;
            assert_eq!(poly_mul_s1.eval(&x), mul(&fx, &s1));
            assert_eq!(poly_mul_s2.eval(&x), mul(&fx, &s2));

            // test inplace add
            f += &g;
            assert_eq!(f.eval(&x), add(&fx, &gx));
        }

        #[test]
        fn phase3_test_poly_monomials() {
            setup();

            // testing add/sub monomials inplace
            let zero = $field::from_u32(0);
            let one = $field::from_u32(1);
            let two = $field::from_u32(2);
            let three = $field::from_u32(3);

            // f(x) = 1+2x^2
            let coeffs = [one, zero, two];
            let mut f = Poly::from_coeffs(HostSlice::from_slice(&coeffs), coeffs.len());
            let x = rand();
            let fx = f.eval(&x);

            f.add_monomial_inplace(&three, 1); // +3x
            let fx_add = f.eval(&x);
            assert_eq!(fx_add, add(&fx, &mul(&three, &x)));

            f.sub_monomial_inplace(&one, 0); // -1
            let fx_sub = f.eval(&x);
            assert_eq!(fx_sub, sub(&fx_add, &one));
        }

        #[test]
        fn phase3_test_poly_read_coeffs() {
            setup();

            let zero = $field::from_u32(0);
            let one = $field::from_u32(1);
            let two = $field::from_u32(2);
            let three = $field::from_u32(3);
            let four = $field::from_u32(4);

            let coeffs = [one, two, three, four];
            let mut f = Poly::from_coeffs(HostSlice::from_slice(&coeffs), coeffs.len());

            // read coeffs to host memory
            let mut host_mem = vec![$field::zero(); coeffs.len()];
            f.copy_coeffs(0, HostSlice::from_mut_slice(&mut host_mem));
            assert_eq!(host_mem, coeffs);

            // read coeffs to device memory
            let mut device_mem = DeviceVec::<$field>::device_malloc(coeffs.len()).unwrap();
            f.copy_coeffs(0, &mut device_mem[..]);
            let mut host_coeffs_from_dev = vec![ScalarField::zero(); coeffs.len() as usize];
            device_mem
                .copy_to_host(HostSlice::from_mut_slice(&mut host_coeffs_from_dev))
                .unwrap();

            assert_eq!(host_mem, host_coeffs_from_dev);

            // multiply by two and read single coeff
            f = &f * &two;
            // read single coeff
            let x_squared_coeff = f.get_coeff(2);
            assert_eq!(x_squared_coeff, mul(&two, &three));
        }

        #[test]
        fn phase3_test_poly_division() {
            setup();

            // divide f(x)/g(x), compute q(x), r(x) and check f(x)=q(x)*g(x)+r(x)

            let f = randomize_poly(1 << 12);
            let g = randomize_poly(1 << 4);

            let (q, r) = f.divide(&g);

            let f_reconstructed = &(&q * &g) + &r;
            let x = rand();

            assert_eq!(f.eval(&x), f_reconstructed.eval(&x));
        }

        #[test]
        fn phase3_test_poly_divide_by_vanishing() {
            setup();

            let zero = $field::from_u32(0);
            let one = $field::from_u32(1);
            let minus_one = sub(&zero, &one);
            // compute random f(x) and compute f(x)*v(x) for v(x) vanishing poly
            // divide by vanishing and check that f(x) is reconstructed

            let f = randomize_poly(1 << 12);
            let v_coeffs = [minus_one, zero, zero, zero, one]; // x^4-1
            let v = Poly::from_coeffs(HostSlice::from_slice(&v_coeffs), v_coeffs.len());

            let fv = &f * &v;
            let deg_f = f.degree();
            let deg_fv = fv.degree();
            assert_eq!(deg_f + 4, deg_fv);

            let f_reconstructed = fv.div_by_vanishing(4);
            assert_eq!(deg_f, f_reconstructed.degree());

            let x = rand();
            assert_eq!(f.eval(&x), f_reconstructed.eval(&x));
        }

        #[test]
        fn phase3_test_poly_eval_on_domain() {
            setup();

            let one = $field::from_u32(1);
            let two = $field::from_u32(2);
            let three = $field::from_u32(3);

            let f = randomize_poly(1 << 12);
            let domain = [one, two, three];

            // evaluate to host memory
            let mut host_evals = vec![ScalarField::zero(); domain.len()];
            f.eval_on_domain(
                HostSlice::from_slice(&domain),
                HostSlice::from_mut_slice(&mut host_evals),
            );

            // check eval on domain agrees with eval() method
            assert_eq!(f.eval(&one), host_evals[0]);
            assert_eq!(f.eval(&two), host_evals[1]);
            assert_eq!(f.eval(&three), host_evals[2]);

            // evaluate to device memory
            let mut device_evals = DeviceVec::<ScalarField>::device_malloc(domain.len()).unwrap();
            f.eval_on_domain(HostSlice::from_slice(&domain), &mut device_evals[..]);
            let mut host_evals_from_device = vec![ScalarField::zero(); domain.len()];
            device_evals
                .copy_to_host(HostSlice::from_mut_slice(&mut host_evals_from_device))
                .unwrap();

            // check that evaluation to device memory is equivalent
            assert_eq!(host_evals, host_evals_from_device);

            // use evals as domain (on device) and evaluate from device to host
            f.eval_on_domain(&mut device_evals[..], HostSlice::from_mut_slice(&mut host_evals));
            // check that the evaluations are correct
            assert_eq!(f.eval(&host_evals_from_device[0]), host_evals[0]);
            assert_eq!(f.eval(&host_evals_from_device[1]), host_evals[1]);
            assert_eq!(f.eval(&host_evals_from_device[2]), host_evals[2]);
        }

        #[test]
        fn phase3_test_eval_on_rou_domain() {
            setup();

            let poly_log_size = 10;
            let domain_log_size = poly_log_size + 2; // interpolate 4 times
            let f = randomize_poly(1 << poly_log_size);

            // evaluate f on rou domain of size 4n
            let mut device_evals = DeviceVec::<ScalarField>::device_malloc(1 << domain_log_size).unwrap();
            f.eval_on_rou_domain(domain_log_size, &mut device_evals[..]);

            // construct g from f's evals and assert they are equal
            let g = Poly::from_rou_evals(&device_evals[..], 1 << domain_log_size);
            let diff = &f - &g;
            assert_eq!(diff.degree(), -1); // diff is the zero poly
        }

        #[test]
        fn phase3_test_odd_even_slicing() {
            setup();
            let size = (1 << 10) - 3;
            // slicing even and odd parts and checking
            let f = randomize_poly(size);
            let x = rand();

            let even = f.even();
            let odd = f.odd();
            assert_eq!(f.degree(), even.degree() + odd.degree() + 1);

            // computing even(x) and odd(x) directly
            let expected_even = (0..=f.degree())
                .filter(|&i| i % 2 == 0)
                .rev()
                .fold($field::zero(), |acc, i| {
                    add(&mul(&acc, &x), &f.get_coeff(i as u64))
                });

            let expected_odd = (0..=f.degree())
                .filter(|&i| i % 2 != 0)
                .rev()
                .fold($field::zero(), |acc, i| {
                    add(&mul(&acc, &x), &f.get_coeff(i as u64))
                });

            // check that even(x) and odd(x) compute correctly

            let evenx = even.eval(&x);
            let oddx = odd.eval(&x);
            assert_eq!(expected_even, evenx);
            assert_eq!(expected_odd, oddx);
        }

        use icicle_core::ntt::{ntt, ntt_inplace, NTTConfig, NTTDir, Ordering};

        #[test]
        fn phase3_test_coeffs_slice() {
            setup();

            let size = 4;
            let coeffs = randomize_coeffs::<$field>(size);
            let mut f = Poly::from_coeffs(HostSlice::from_slice(&coeffs), size);

            // take a mutable coeffs slice as a DeviceSlice
            let coeffs_slice_dev = f.coeffs_mut_slice();
            assert_eq!(coeffs_slice_dev.len(), size);
            assert!(coeffs_slice_dev.is_on_device());

            // let g = &f + &f; // cannot borrow here since s is a mutable slice of f

            // copy to host and check equality
            let mut coeffs_copied_from_slice = vec![ScalarField::zero(); coeffs_slice_dev.len()];
            coeffs_slice_dev
                .copy_to_host(HostSlice::from_mut_slice(&mut coeffs_copied_from_slice))
                .unwrap();
            assert_eq!(coeffs_copied_from_slice, coeffs);

            // or can use the memory directly
            let mut config: NTTConfig<$field> = NTTConfig::default();
            let mut ntt_result = vec![$field::zero(); coeffs_slice_dev.len()];
            ntt(
                coeffs_slice_dev,
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_result),
            )
            .unwrap();
            // ntt[0] is f(one) because it's the sum of coeffs
            assert_eq!(ntt_result[0], f.eval(&$field::one()));

            // after last use of coeffs_slice_dev, can borrow f again
            let g = &f * &f;
            assert_eq!(mul(&ntt_result[0], &ntt_result[0]), g.eval(&$field::one()));
        }
    };
}
