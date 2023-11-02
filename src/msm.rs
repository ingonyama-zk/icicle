use std::ffi::c_uint;

use crate::{cuda::*, curve::*};

/*
/**
 * @struct MSMConfig
 * Struct that encodes MSM parameters to be passed into the [msm](@ref msm) function.
 */
struct MSMConfig {
  bool are_scalars_on_device;         /**< True if scalars are on device and false if they're on host. Default value: false. */
  bool are_scalars_montgomery_form;   /**< True if scalars are in Montgomery form and false otherwise. Default value: true. */
  unsigned points_size;               /**< Number of points in the MSM. If a batch of MSMs needs to be computed, this should be a number
                                       *   of different points. So, if each MSM re-uses the same set of points, this variable is set equal
                                       *   to the MSM size. And if every MSM uses a distinct set of points, it should be set to the product of
                                       *   MSM size and [batch_size](@ref batch_size). Default value: 0 (meaning it's equal to the MSM size). */
  unsigned precompute_factor;         /**< The number of extra points to pre-compute for each point. Larger values decrease the number of computations
                                       *   to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute). */
  bool are_points_on_device;          /**< True if points are on device and false if they're on host. Default value: false. */
  bool are_points_montgomery_form;    /**< True if coordinates of points are in Montgomery form and false otherwise. Default value: true. */
  unsigned batch_size;                /**< The number of MSMs to compute. Default value: 1. */
  bool are_result_on_device;          /**< True if the results should be on device and false if they should be on host. Default value: false. */
  unsigned c;                         /**< \f$ c \f$ value, or "window bitsize" which is the main parameter of the "bucket method"
                                       *   that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
                                       *   footprint but also more parallelism and less computational complexity (up to a certain point).
                                       *   Default value: 0 (the optimal value of \f$ c \f$ is chosen automatically). */
  unsigned bitsize;                   /**< Number of bits of the largest scalar. Typically equals the bitsize of scalar field, but if a different
                                       *   (better) upper bound is known, it should be reflected in this variable. Default value: 0 (set to the bitsize of scalar field). */
  bool big_triangle;                  /**< Whether to do "bucket accumulation" serially. Decreases computational complexity, but also greatly
                                       *   decreases parallelism, so only suitable for large batches of MSMs. Default value: false. */
  unsigned large_bucket_factor;       /**< Variable that controls how sensitive the algorithm is to the buckets that occur very frequently.
                                       *   Useful for efficient treatment of non-uniform distributions of scalars and "top windows" with few bits.
                                       *   Can be set to 0 to disable separate treatment of large buckets altogether. Default value: 10. */
  device_context::DeviceContext ctx;  /**< Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext). */
};
*/
/// Struct that encodes MSM parameters to be passed into the `msm` function.

#[repr(C)]
pub struct MSMConfig {
    /// True if scalars are on device and false if they're on host. Default value: false.
    pub are_scalars_on_device: bool,

    /// True if scalars are in Montgomery form and false otherwise. Default value: true.
    pub are_scalars_montgomery_form: bool,

    /// Number of points in the MSM. If a batch of MSMs needs to be computed, this should be a number
    /// of different points. So, if each MSM re-uses the same set of points, this variable is set equal
    /// to the MSM size. And if every MSM uses a distinct set of points, it should be set to the product of
    /// MSM size and batch_size. Default value: 0 (meaning it's equal to the MSM size).
    pub points_size: usize, // Note: `unsigned` in C++ corresponds to `u32` in Rust

    /// The number of extra points to pre-compute for each point. Larger values decrease the number of computations
    /// to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute).
    pub precompute_factor: usize,

    /// True if points are on device and false if they're on host. Default value: false.
    pub are_points_on_device: bool,

    /// True if coordinates of points are in Montgomery form and false otherwise. Default value: true.
    pub are_points_montgomery_form: bool,

    /// The number of MSMs to compute. Default value: 1.
    pub batch_size: usize,

    /// True if the results should be on device and false if they should be on host. Default value: false.
    pub are_result_on_device: bool,

    /// `c` value, or "window bitsize" which is the main parameter of the "bucket method"
    /// that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
    /// footprint but also more parallelism and less computational complexity (up to a certain point).
    /// Default value: 0 (the optimal value of `c` is chosen automatically).
    pub c: usize,

    /// Number of bits of the largest scalar. Typically equals the bitsize of scalar field, but if a different
    /// (better) upper bound is known, it should be reflected in this variable. Default value: 0 (set to the bitsize of scalar field).
    pub bitsize: usize,

    /// Whether to do "bucket accumulation" serially. Decreases computational complexity, but also greatly
    /// decreases parallelism, so only suitable for large batches of MSMs. Default value: false.
    pub is_big_triangle: bool,

    /// Variable that controls how sensitive the algorithm is to the buckets that occur very frequently.
    /// Useful for efficient treatment of non-uniform distributions of scalars and "top windows" with few bits.
    /// Can be set to 0 to disable separate treatment of large buckets altogether. Default value: 10.
    pub large_bucket_factor: usize,

    /// Details related to the device such as its id and stream id.
    pub ctx: DeviceContext,
}

extern "C" {
    #[link_name = "bn254MSMCuda"]
    fn msm_cuda(
        scalars: *const ScalarField,
        points: *const PointAffineNoInfinity,
        count: usize,
        config: MSMConfig,
        out: *mut Point,
    ) -> c_uint;

    // #[link_name = "GetDefaultMSMConfig"]
    fn GetDefaultMSMConfig() -> MSMConfig;
}

pub fn get_default_msm_config() -> MSMConfig {
    unsafe { GetDefaultMSMConfig() }
}

pub fn msm(scalars: &[ScalarField], points: &[PointAffineNoInfinity]) -> Point {
    let count = points.len();
    if count != scalars.len() {
        todo!("variable length")
    }

    let mut out = Point::zero();
    unsafe {
        msm_cuda(
            scalars as *const _ as *const ScalarField,
            points as *const _ as *const PointAffineNoInfinity,
            points.len(),
            get_default_msm_config(),
            &mut out as *mut _ as *mut Point,
        )
    };

    out
}

#[cfg(test)]
pub(crate) mod tests {
    use ark_bn254::{Fr, G1Affine, G1Projective};
    // use ark_bls12_381::{Fr, G1Projective};
    use ark_ec::msm::VariableBaseMSM;
    use ark_ff::PrimeField;
    use ark_std::UniformRand;
    use rand::RngCore;

    use crate::{curve::*, msm::*, utils::get_rng};

    const SLICE_LEN: usize = 100;

    pub fn generate_random_points(count: usize, mut rng: Box<dyn RngCore>) -> Vec<PointAffineNoInfinity> {
        (0..SLICE_LEN)
            .map(|_| PointAffineNoInfinity::from_ark(&G1Affine::from(G1Projective::rand(&mut rng))))
            .collect::<Vec<_>>()
            .into_iter()
            .cycle()
            .take(count)
            .collect()
    }

    #[allow(dead_code)]
    pub fn generate_random_points_proj(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Point> {
        (0..SLICE_LEN)
            .map(|_| Point::from_ark(G1Projective::rand(&mut rng)))
            .collect::<Vec<_>>()
            .into_iter()
            .cycle()
            .take(count)
            .collect()
    }

    pub fn generate_random_scalars(count: usize, mut rng: Box<dyn RngCore>) -> Vec<ScalarField> {
        (0..count)
            .map(|_| ScalarField::from_ark(Fr::rand(&mut rng).into_repr()))
            .collect()
    }

    #[test]
    // #[ignore = "skip"]
    fn test_msm() {
        let test_sizes = [24];

        for pow2 in test_sizes {
            let count = 1 << pow2;
            let seed = None; // set Some to provide seed
            let points = generate_random_points(count, get_rng(seed));
            let scalars = generate_random_scalars(count, get_rng(seed));

            let msm_result = msm(&scalars, &points);

            let point_r_ark: Vec<_> = points
                .iter()
                .map(|x| x.to_ark_repr())
                .collect();
            let scalars_r_ark: Vec<_> = scalars
                .iter()
                .map(|x| x.to_ark())
                .collect();

            let msm_result_ark = VariableBaseMSM::multi_scalar_mul(&point_r_ark, &scalars_r_ark);

            assert_eq!(msm_result.to_ark_affine(), msm_result_ark);
            assert_eq!(msm_result.to_ark(), msm_result_ark);
            assert_eq!(
                msm_result.to_ark_affine(),
                Point::from_ark(msm_result_ark).to_ark_affine()
            );
        }
    }
}
