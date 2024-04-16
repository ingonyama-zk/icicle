use crate::curve::ScalarField;
use icicle_core::impl_polynomial_api;

impl_polynomial_api!("bn254", bn254, ScalarField);

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::FftField;
    use icicle_core::ntt::{initialize_domain, release_domain, NTTDomain};
    use icicle_core::traits::ArkConvertible;
    use icicle_cuda_runtime::device_context::DeviceContext;
    use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};

    use icicle_core::traits::FieldImpl;

    fn init_domain<F: FieldImpl + ArkConvertible>(max_size: u64, device_id: usize, fast_twiddles_mode: bool)
    where
        F::ArkEquivalent: FftField,
        <F as FieldImpl>::Config: NTTDomain<F>,
    {
        let ctx = DeviceContext::default_for_device(device_id);
        let ark_rou = F::ArkEquivalent::get_root_of_unity(max_size).unwrap();
        initialize_domain(F::from_ark(ark_rou), &ctx, fast_twiddles_mode).unwrap();
    }

    fn rel_domain<F: FieldImpl>(device_id: usize)
    where
        <F as FieldImpl>::Config: NTTDomain<F>,
    {
        let ctx = DeviceContext::default_for_device(device_id);
        release_domain::<F>(&ctx).unwrap()
    }

    // TODO Yuval: implement small tests
    #[test]
    fn test_new_polynomial() {
        let device_id: usize = 0;
        let domain_max_size: u64 = 1 << 16;
        init_domain::<ScalarField>(domain_max_size, device_id, false /*=fast twiddle */);

        bn254::Polynomial::init_cuda_backend();

        let size: usize = 2;
        let coeffs = [ScalarField::zero(), ScalarField::one()];
        let evals = [ScalarField::one(), ScalarField::one()];

        let mut f = bn254::Polynomial::from_coeffs(HostSlice::from_slice(&coeffs), size);
        let g = bn254::Polynomial::from_rou_evals(HostSlice::from_slice(&evals), size);

        let h = f.clone();
        f.print();
        g.print();
        h.print();
        let sum = &(&h + &g) + &f;
        h.print();
        sum.print();
        f += &sum;
        f.print();
        sum.print();
        let sub = &f - &g;
        sub.print();
        let mul = &sum * &h;
        mul.print();
        let two = ScalarField::from_u32(2);
        let three = ScalarField::from_u32(3);
        let mul_scalar = &(&(&two * &f) * &three) * &two;
        f.print();
        mul_scalar.print();

        let q = &mul_scalar / &sum;
        let r = &mul_scalar % &sum;
        q.print();
        r.print();

        let (q2, r2) = mul_scalar.divide(&sum);
        q2.print();
        r2.print();

        let mut new_mul_scalar = &(&q * &sum) + &r;
        new_mul_scalar.print();

        let div_by_v = new_mul_scalar.div_by_vanishing(2);
        div_by_v.print();

        new_mul_scalar.add_monomial_inplace(&two, 2);
        new_mul_scalar.print();
        new_mul_scalar.sub_monomial_inplace(&three, 1);
        new_mul_scalar.print();

        println!("new_mul_scalar(2) = {}", new_mul_scalar.eval(&two));

        let domain = [
            ScalarField::from_u32(1),
            ScalarField::from_u32(2),
            ScalarField::from_u32(3),
        ];

        let mut evals = vec![ScalarField::zero(); domain.len()];
        new_mul_scalar.eval_on_domain(HostSlice::from_slice(&domain), HostSlice::from_mut_slice(&mut evals));
        println!("degree = {}", new_mul_scalar.degree());

        let mut device_evals = DeviceVec::<ScalarField>::cuda_malloc(domain.len()).unwrap();
        new_mul_scalar.eval_on_domain(HostSlice::from_slice(&domain), &mut device_evals[..]);
        let mut host_evals_from_device = vec![ScalarField::zero(); domain.len()];
        device_evals
            .copy_to_host(HostSlice::from_mut_slice(&mut host_evals_from_device))
            .unwrap();
        println!("evals on domain: {:?}", &evals);
        println!("(from device) evals on domain = {:?}", host_evals_from_device);

        println!("coeff[2] = {}", new_mul_scalar.get_coeff(1));

        let mut host_coeffs = vec![ScalarField::zero(); 3 as usize];
        new_mul_scalar.copy_coeffs(0, HostSlice::from_mut_slice(&mut host_coeffs));
        println!("coeffs = {:?}", host_coeffs);

        let mut device_coeffs = DeviceVec::<ScalarField>::cuda_malloc(3).unwrap();
        new_mul_scalar.copy_coeffs(0, &mut device_coeffs[..]);
        let mut host_coeffs_from_dev = vec![ScalarField::zero(); 3 as usize];
        device_coeffs
            .copy_to_host(HostSlice::from_mut_slice(&mut host_coeffs_from_dev))
            .unwrap();
        println!("coeffs_from_dev = {:?}", host_coeffs_from_dev);

        rel_domain::<ScalarField>(device_id);
    }
}
