pub mod field;
mod matrix;
mod utils;

pub const FLOW_SIZE: usize = 1 << 12; //4096  //prod flow size
pub const TEST_SIZE_DIV: usize = 8; //TODO: Prod size / test size for speedup
pub const TEST_SIZE: usize = FLOW_SIZE / TEST_SIZE_DIV; //test flow size
pub const M_POINTS: usize = TEST_SIZE; //TODO: 4096
pub const SRS_SIZE: usize = M_POINTS; //TODO: 4096
pub const S_GROUP_SIZE: usize = 2 * M_POINTS; //TODO: 8192
pub const N_ROWS: usize = 256 / TEST_SIZE_DIV;
pub const FOLD_SIZE: usize = 512 / TEST_SIZE_DIV;

use std::ffi::{c_int, c_uint, c_void};

use ark_bls12_381::Fr;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use field::{
    generate_random_points, generate_random_scalars, Point, PointAffineNoInfinity, Scalar,
    ScalarField,
};
use utils::{export_limbs, get_rng};

use crate::{
    field::SCALAR_LIMBS,
    matrix::split_vec_to_matrix,
    utils::{csv_to_u32_limbs, from_limbs, import_limbs},
};

// TODO:
// Implement Point functions:
// New, Add , Mult, from_Affine, to_affine -> prints point, output is 2* BigUint, to_Mont, from_Mont,
//
// Implement Scalar functions:
// New, Add, Mult
//

fn get_debug_data(
    filename: &str,
    limbsize: usize,
    height: usize,
    lenght: usize,
) -> Vec<Vec<Scalar>> {
    let data_root_path = "../data/test_vectors/";

    let limbs = csv_to_u32_limbs(&format!("{}{}", data_root_path, filename), limbsize);

    let result = split_vec_to_matrix(&from_limbs(limbs, limbsize, Scalar::from_limbs), lenght);
    assert_eq!(result.len(), height);
    assert_eq!(result[0].len(), lenght);

    result
}
extern "C" {
    fn msm_cuda(
        out: *mut Point,
        points: *const PointAffineNoInfinity,
        scalars: *const ScalarField,
        count: usize, //TODO: is needed?
        device_id: usize,
    ) -> c_uint;
}

pub fn msm(points: &[PointAffineNoInfinity], scalars: &[Scalar], device_id: usize) -> Point {
    let count = points.len();
    if count != scalars.len() {
        todo!("variable length")
    }

    let mut ret = Point::zero();
    unsafe {
        msm_cuda(
            &mut ret as *mut _ as *mut Point,
            points as *const _ as *const PointAffineNoInfinity,
            scalars as *const _ as *const ScalarField,
            scalars.len(),
            device_id,
        )
    };

    ret
}

///
/// NTT
///

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NTTType {
    Std = 0,
    Coset = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NTTDir {
    Fwd = 0, //TODO: naming?
    Inv = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NTTInOutOrd {
    NN = 0,
    NR = 1,
    RN = 2,
    RR = 3,
}

extern "C" {
    fn ntt_end2end(inout: *mut ScalarField, n: usize, inverse: bool) -> c_int;

    fn ecntt_end2end(inout: *mut Point, n: usize, inverse: bool) -> c_int;

    fn vec_mod_mult_point(
        inout: *mut Point,
        scalars: *const ScalarField,
        n_elemes: usize,
        device_id: usize,
    ) -> c_int;

    fn vec_mod_mult_scalar(
        inout: *mut ScalarField,
        scalars: *const ScalarField,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;
}

pub fn ntt(values: &mut [Scalar], device_id: usize) {
    ntt_internal(values, NTTDir::Fwd, device_id);
}

pub fn intt(values: &mut [Scalar], device_id: usize) {
    ntt_internal(values, NTTDir::Inv, device_id);
}

/// Compute an in-place NTT on the input data.
fn ntt_internal(v: &mut [Scalar], direction: NTTDir, device_id: usize) -> i32 {
    let len = v.len();
    let lg_domain_size = (len as f32).log2().ceil() as u32 + 0;

    let domain_size = 1usize << lg_domain_size;

    let domain = GeneralEvaluationDomain::<Fr>::new(domain_size).unwrap();

    let arc_values = v.iter().map(|v| v.to_ark()).collect::<Vec<Fr>>();
    let mut vs = arc_values.clone();

    if direction == NTTDir::Fwd {
        domain.fft_in_place(&mut vs);
    } else if direction == NTTDir::Inv {
        domain.ifft_in_place(&mut vs);
    }

    reorder(&mut vs);

    for i in 0..len {
        //println!("{}", vs[i].0.to_string());
        let ark = Scalar::from_ark(vs[i]);
        //println!("{:08X?}", ark);
        v[i] = ark
    }

    return 0;
}

fn reorder<T: Copy>(v: &mut [T]) {
    let len = v.len();

    //let n = 170;
    // (domain_size - 2)/3;
    // let temp = v[2 * n + 1];
    // v[2 * n + 1] = v[1];
    // v[1] = temp;
    let vcl = v.to_vec();
    let ind = if len == 512 {
        vec![
            0, 509, 506, 503, 500, 497, 494, 491, 488, 485, 482, 479, 476, 473, 470, 467, 464, 461,
            458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 425, 422, 419, 416, 413, 410,
            407, 404, 401, 398, 395, 392, 389, 386, 383, 380, 377, 374, 371, 368, 365, 362, 359,
            356, 353, 350, 347, 344, 341, 338, 335, 332, 329, 326, 323, 320, 317, 314, 311, 308,
            305, 302, 299, 296, 293, 290, 287, 284, 281, 278, 275, 272, 269, 266, 263, 260, 257,
            254, 251, 248, 245, 242, 239, 236, 233, 230, 227, 224, 221, 218, 215, 212, 209, 206,
            203, 200, 197, 194, 191, 188, 185, 182, 179, 176, 173, 170, 167, 164, 161, 158, 155,
            152, 149, 146, 143, 140, 137, 134, 131, 128, 125, 122, 119, 116, 113, 110, 107, 104,
            101, 98, 95, 92, 89, 86, 83, 80, 77, 74, 71, 68, 65, 62, 59, 56, 53, 50, 47, 44, 41,
            38, 35, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 511, 508, 505, 502, 499, 496, 493,
            490, 487, 484, 481, 478, 475, 472, 469, 466, 463, 460, 457, 454, 451, 448, 445, 442,
            439, 436, 433, 430, 427, 424, 421, 418, 415, 412, 409, 406, 403, 400, 397, 394, 391,
            388, 385, 382, 379, 376, 373, 370, 367, 364, 361, 358, 355, 352, 349, 346, 343, 340,
            337, 334, 331, 328, 325, 322, 319, 316, 313, 310, 307, 304, 301, 298, 295, 292, 289,
            286, 283, 280, 277, 274, 271, 268, 265, 262, 259, 256, 253, 250, 247, 244, 241, 238,
            235, 232, 229, 226, 223, 220, 217, 214, 211, 208, 205, 202, 199, 196, 193, 190, 187,
            184, 181, 178, 175, 172, 169, 166, 163, 160, 157, 154, 151, 148, 145, 142, 139, 136,
            133, 130, 127, 124, 121, 118, 115, 112, 109, 106, 103, 100, 97, 94, 91, 88, 85, 82, 79,
            76, 73, 70, 67, 64, 61, 58, 55, 52, 49, 46, 43, 40, 37, 34, 31, 28, 25, 22, 19, 16, 13,
            10, 7, 4, 1, 510, 507, 504, 501, 498, 495, 492, 489, 486, 483, 480, 477, 474, 471, 468,
            465, 462, 459, 456, 453, 450, 447, 444, 441, 438, 435, 432, 429, 426, 423, 420, 417,
            414, 411, 408, 405, 402, 399, 396, 393, 390, 387, 384, 381, 378, 375, 372, 369, 366,
            363, 360, 357, 354, 351, 348, 345, 342, 339, 336, 333, 330, 327, 324, 321, 318, 315,
            312, 309, 306, 303, 300, 297, 294, 291, 288, 285, 282, 279, 276, 273, 270, 267, 264,
            261, 258, 255, 252, 249, 246, 243, 240, 237, 234, 231, 228, 225, 222, 219, 216, 213,
            210, 207, 204, 201, 198, 195, 192, 189, 186, 183, 180, 177, 174, 171, 168, 165, 162,
            159, 156, 153, 150, 147, 144, 141, 138, 135, 132, 129, 126, 123, 120, 117, 114, 111,
            108, 105, 102, 99, 96, 93, 90, 87, 84, 81, 78, 75, 72, 69, 66, 63, 60, 57, 54, 51, 48,
            45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3,
        ]
    } else if len == 32 {
        vec![
            0, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 30,
            27, 24, 21, 18, 15, 12, 9, 6, 3,
        ]
    } else {
        panic!("unsupported length: {}", len)
    };

    for i in 0..len {
        v[ind[i]] = vcl[i];
    }
    // let mut shift = vec![999; 512];
    // let n = 170;
    // shift[0] = 0;
    // shift[2 * n + 1] = 1;

    // for i in 0..n {
    //     shift[i + 1] = n * 3 - i * 3 - 1;
    //     shift[n + i + 1] = n * 3 - i * 3 + 1;
    //     shift[2 * n + i + 2] = n * 3 - i * 3;
    // }

    // for i in 0..n {
    //     let j = n * 3 - i * 3;
    //     let j1 = j - 1;
    //     let j2 = j + 1;

    //     let k = i + 1;
    //     let temp = v[k];
    //     v[k] = v[j1];
    //     v[j1] = temp;

    //     let k2 = n + i + 1;
    //     let temp = v[k2];
    //     v[k2] = v[j2];
    //     v[j2] = temp;
    //     let k3 = 2 * n + i + 2;
    //     let temp = v[k3];
    //    Invalid v[k3] = v[j];
    //     v[j] = temp;
    // }
}

/// Compute an in-place ECNTT on the input data.
fn ecntt_internal(
    values: &mut [Point],
    direction: NTTDir,
    device_id: usize,
    is_naive: u32, // TODO: for testing only
) -> i32 {
    return 0;
    // unsafe {
    //     ecntt_end2end(
    //         values.as_mut_ptr() as *mut Point,
    //         values.len(),
    //         inverse,
    //     )
    // }
}

pub fn ecntt(values: &mut [Point], device_id: usize) {
    ecntt_internal(values, NTTDir::Fwd, device_id, 0);
}

/// Compute an in-place iECNTT on the input data.
pub fn iecntt(values: &mut [Point], device_id: usize) {
    ecntt_internal(values, NTTDir::Inv, device_id, 0);
}

fn store_random_scalars(
    seed: Option<u64>,
    scnt: usize,
    file_path: &str,
) -> (usize, Vec<Scalar>, Vec<u32>) {
    let rng = get_rng(seed);

    let sc_len = Scalar::default().limbs().len();
    let scalars = generate_random_scalars(scnt, rng);

    let limbs = scalars
        .iter()
        .map(|sc| sc.limbs())
        .flatten()
        .collect::<Vec<_>>();
    export_limbs(&limbs, file_path);
    (sc_len, scalars, limbs)
}

fn store_random_points(
    //TODO: generic
    seed: Option<u64>,
    cnt: usize,
    file_path: &str,
) -> (usize, Vec<PointAffineNoInfinity>, Vec<u32>) {
    let rng = get_rng(seed);

    let pa_len = PointAffineNoInfinity::default().limbs().len();
    let points = generate_random_points(cnt, rng);

    let limbs = points
        .iter()
        .map(|p| p.limbs())
        .flatten()
        .collect::<Vec<_>>();
    export_limbs(&limbs, file_path);
    (pa_len, points, limbs)
}

pub fn import_scalars(filepath: &str, expected_count: usize) -> Vec<Scalar> {
    store_random_scalars(Some(4), expected_count, filepath); //TODO: disable

    let limbs = import_limbs(filepath);

    let scalars = limbs
        .chunks(SCALAR_LIMBS)
        .map(|lmbs| Scalar::from_limbs(lmbs))
        .collect::<Vec<Scalar>>();

    assert_eq!(scalars.len(), expected_count);
    scalars
}

pub fn multp_vec(a: &mut [Point], b: &[Scalar], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_point(
            a as *mut _ as *mut Point,
            b as *const _ as *const ScalarField,
            a.len(),
            device_id,
        );
    }
}

pub fn mult_sc_vec(a: &mut [Scalar], b: &[Scalar], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_scalar(
            a as *mut _ as *mut ScalarField,
            b as *const _ as *const ScalarField,
            a.len(),
            device_id,
        );
    }
}


#[cfg(test)]
mod tests {
    use std::{mem::transmute_copy, vec};

    use ark_bls12_381::{Fq, Fr, G1Projective};
    use ark_ff::{BigInteger256, BigInteger384};

    use crate::{
        field::{generate_random_points, BaseField, BASE_LIMBS, SCALAR_LIMBS},
        matrix::{rows_to_cols, split_vec_to_matrix},
        utils::{hex_str_be_to_u32_vec, import_limbs},
        *,
    };
    use std::ops::{BitAnd, Sub};
    use utils::get_fixed_limbs;

    fn vec_u32_to_u64<const N: usize>(v: &[u32]) -> [u64; N] {
        let vu64 = v
            .chunks(2)
            .map(|x| {
                u64::from_le_bytes(
                    [x[0].to_le_bytes(), x[1].to_le_bytes()]
                        .concat()
                        .try_into()
                        .unwrap(),
                )
            })
            .collect::<Vec<u64>>()
            .try_into()
            .unwrap();
        vu64
    }

    #[test]
    fn test_msm() {
        let seed = None; //some value to fix the rng

        let points = generate_random_points(1 << 10, get_rng(seed));
        let scalars = generate_random_scalars(1 << 10, get_rng(seed));

        let msm_result = msm(&points, &scalars, 0);

        assert_eq!(
            msm_result.y,
            BaseField {
                s: get_fixed_limbs(&[1u32])
            }
        );
    }

    #[test]
    fn test_ntt() {
        let seed = None; //some value to fix the rng
        let test_size = 1 << 4;

        let points = generate_random_points(test_size, get_rng(seed));
        let scalars = generate_random_scalars(test_size, get_rng(seed));

        let mut naive_ntt_result = scalars.clone();
        ntt_internal(&mut naive_ntt_result, NTTDir::Fwd, 0);

        let mut ntt_result = scalars.clone();
        ntt(&mut ntt_result, 0);

        assert_eq!(ntt_result, naive_ntt_result);

        let mut intt_result = ntt_result.clone();

        intt(&mut intt_result, 0);

        assert_eq!(intt_result, scalars);

        let points_proj = points.iter().map(|p| p.to_projective()).collect::<Vec<_>>();

        let mut naive_ecntt_result = points_proj.clone();
        ecntt_internal(&mut naive_ecntt_result, NTTDir::Fwd, 0, 1);

        let mut ecntt_result = points_proj.clone();
        ecntt(&mut ecntt_result, 0);

        assert_eq!(ecntt_result, naive_ecntt_result);

        let mut iecntt_result = ecntt_result.clone();
        iecntt(&mut iecntt_result, 0);

        assert_eq!(iecntt_result, points_proj);
    }

    #[test]
    fn test_read_write_scalars() {
        //two scalars
        //
        // let scalars = [
        //     0x010000f1, 0x020000f2, 0x030000f3, 0x040000f4, 0x050000f5, 0x060000f6, 0x070000f7,
        //     0x080000f8, //
        //     0x210000f1, 0x220000f2, 0x230000f3, 0x240000f4, 0x250000f5, 0x260000f6, 0x270000f7,
        //     0x280000f8,
        // ]
        // .chunks(sc_len)
        // .map(|lmbs| Scalar::from_limbs(lmbs))
        // .collect::<Vec<Scalar>>();

        let scnt = 512;
        let file_path = "omegas1.bin";
        let seed = Some(1);

        let (sc_len, scalars, limbs) = store_random_scalars(seed, scnt, file_path);

        let result = import_limbs(file_path);
        assert_eq!(result, limbs);

        let scalars_from_limbs = result
            .chunks(sc_len)
            .map(|lmbs| Scalar::from_limbs(lmbs))
            .collect::<Vec<Scalar>>();

        assert_eq!(scalars_from_limbs, scalars);
    }

    #[test]
    fn test_read_write_points() {
        let field_size2x = BaseField::default().limbs().len() * 2;
        // // two points [x,y]
        // let points = [
        //     0x010000f1, 0x020000f2, 0x030000f3, 0x040000f4, 0x050000f5, 0x060000f6, 0x070000f7,
        //     0x080000f8, 0x080000f9, 0x080000fa, 0x080000fb, 0x080000fc,// x1
        //     0x210000f1, 0x220000f2, 0x230000f3, 0x240000f4, 0x250000f5, 0x260000f6, 0x270000f7,
        //     0x280000f8, 0x280000f9, 0x280000fa, 0x280000fb, 0x280000fc,// y1
        //     0x310000f1, 0x320000f2, 0x330000f3, 0x340000f4, 0x350000f5, 0x360000f6, 0x370000f7,
        //     0x380000f8, 0x380000f9, 0x380000fa, 0x380000fb, 0x380000fc,// x2
        //     0x410000f1, 0x420000f2, 0x430000f3, 0x440000f4, 0x450000f5, 0x460000f6, 0x470000f7,
        //     0x480000f8, 0x480000f9, 0x480000fa, 0x480000fb, 0x480000fc,// y2
        // ]
        // .chunks(field_size2x)
        // .map(|lmbs| PointAffineNoInfinity::from_xy_limbs(lmbs))
        // .collect::<Vec<PointAffineNoInfinity>>();

        let count = 512;
        let file_path = "S.bin";
        let seed = Some(1);

        let rng = get_rng(seed);

        let points = generate_random_points(count, rng);

        let limbs = points
            .iter()
            .map(|pa| pa.limbs())
            .flatten()
            .collect::<Vec<_>>();
        export_limbs(&limbs, file_path);

        let result = import_limbs(file_path);
        assert_eq!(result, limbs);

        let points_from_limbs = result
            .chunks(field_size2x)
            .map(|lmbs| PointAffineNoInfinity::from_xy_limbs(lmbs))
            .collect::<Vec<PointAffineNoInfinity>>();

        assert_eq!(points_from_limbs, points);
    }

    #[test]
    fn test_to_from_arc_bls12_381_377() {
        let sc = generate_random_scalars(1, get_rng(Some(1)))[0];
        let vu32: [u32; 8] = sc.limbs().try_into().unwrap();

        let vu64: [u64; SCALAR_LIMBS / 2] = vec_u32_to_u64(&vu32);

        let arc_256 = BigInteger256::new(vu64);
        let f = Fr::new(arc_256);
        let ingo_sc = Scalar::from_limbs(&vu32);
        assert_eq!(arc_256, unsafe { transmute_copy(&ingo_sc) });
        assert_eq!(f, ingo_sc.to_ark());

        let p = generate_random_points(1, get_rng(Some(1)))[0].to_projective();
        let _vu32_xyz: [u32; BASE_LIMBS * 3] = [p.x.limbs(), p.y.limbs(), p.z.limbs()]
            .concat()
            .try_into()
            .unwrap();

        let vu32 = p.x.limbs();

        let vu64_x = vu32
            .chunks(2)
            .map(|s| {
                u64::from_le_bytes(
                    [s[0].to_le_bytes(), s[1].to_le_bytes()]
                        .concat()
                        .try_into()
                        .unwrap(),
                )
            })
            .collect::<Vec<u64>>()
            .try_into()
            .unwrap();

        let arc_384 = BigInteger384::new(vu64_x);
        let ingo_bf = p.x;
        let trans_bf = unsafe { transmute_copy(&ingo_bf) };
        assert_eq!(arc_384, trans_bf, "{:08X?} {:08X?}", arc_384, trans_bf);
        let f = Fq::new(arc_384);
        assert_eq!(f, unsafe { transmute_copy(&ingo_bf) });

        let arc_proj: G1Projective = G1Projective::new(
            unsafe { transmute_copy(&p.x) },
            unsafe { transmute_copy(&p.y) },
            unsafe { transmute_copy(&p.z) },
        );

        assert_eq!(arc_proj, unsafe {
            transmute_copy::<Point, G1Projective>(&p)
        });
    }

    #[test]
    fn test_debug_scalar() {
        let sample = "12039d72be179e9cf6d56ce4a99bd4e8b8ff59367c57bf66f094d4143f7d4ade";
        assert_eq!(64, sample.len());
        let limbs = hex_str_be_to_u32_vec(sample, SCALAR_LIMBS);

        assert_eq!(limbs.len(), SCALAR_LIMBS);
        let sc = Scalar::from_limbs(&limbs);

        let arc_sc = Fr::new(BigInteger256::new(
            vec_u32_to_u64::<4>(&sc.s.limbs()).try_into().unwrap(),
        ));

        assert_eq!(arc_sc, sc.to_ark());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_arc_py_scalar() {
        let py_str = "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";

        let limbs = hex_str_be_to_u32_vec(py_str, SCALAR_LIMBS);

        let sc = Scalar::from_limbs(&limbs);

        let ark_mod = Fr::new(BigInteger256::new([
            0xffffffff00000001,
            0x53bda402fffe5bfe,
            0x3339d80809a1d805,
            0x73eda753299d7d48,
        ]));

        assert_eq!(
            sc.to_ark(),
            ark_mod,
            "\n******\n{:08X?}\n******\n{:08X?}\n******\n",
            sc.to_ark(),
            ark_mod
        );

        let sample = "12039d72be179e9cf6d56ce4a99bd4e8b8ff59367c57bf66f094d4143f7d4ade";
        assert_eq!(64, sample.len());
        let limbs = hex_str_be_to_u32_vec(sample, SCALAR_LIMBS);

        assert_eq!(limbs.len(), SCALAR_LIMBS);
        let sc = Scalar::from_limbs(&limbs);

        let arc_sc = Fr::new(BigInteger256::new(
            vec_u32_to_u64::<4>(&sc.s.limbs()).try_into().unwrap(),
        ));
        assert_eq!(arc_sc, sc.to_ark());
    }

    #[test]
    #[allow(non_snake_case)]

    fn test_debug_ntt() {
        let D_in_limbs = get_scalar_limbs_csv_debug("D_in.csv");

        assert_eq!(D_in_limbs.len(), N_ROWS * M_POINTS * SCALAR_LIMBS);

        let limbs = from_limbs(D_in_limbs, SCALAR_LIMBS, Scalar::from_limbs);

        assert_eq!(limbs.len(), N_ROWS * M_POINTS);

        let D_in = split_vec_to_matrix(&limbs, M_POINTS);
        assert_eq!(D_in.len(), N_ROWS);
        assert_eq!(D_in[0].len(), M_POINTS);

        let mut D_in_row0 = D_in[0].clone();
        intt(&mut D_in_row0, 0);

        let C_rows_limbs = get_scalar_limbs_csv_debug("C_rows.csv");

        let C_rows = split_vec_to_matrix(
            &from_limbs(C_rows_limbs, SCALAR_LIMBS, Scalar::from_limbs),
            M_POINTS,
        );

        assert_eq!(C_rows.len(), N_ROWS);
        assert_eq!(C_rows[0].len(), M_POINTS);

        assert_print(&C_rows[0], D_in_row0);

        let C_debug = get_debug_data("C.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);

        let C_debug_rot = rows_to_cols(&C_debug);

        let C_rows_rot = rows_to_cols(&C_rows);
        let mut c_rows0_rot = C_rows_rot[0].clone();

        intt(&mut c_rows0_rot, 0);
        assert_print(&C_debug_rot[0], c_rows0_rot);

        // C = INTT_cols(C_rows) 256x4096 col
        let C = rows_to_cols(
            &mut (0..M_POINTS)
                .map(|i| {
                    let mut col = C_rows.iter().map(|row| row[i]).collect::<Vec<Scalar>>();
                    intt(&mut col, 0);
                    col
                })
                .collect::<Vec<Vec<_>>>(),
        );
        assert_eq!(C, C_debug);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_saclar_mul() {
        let mut intoo = [Scalar::one(), Scalar::one(), Scalar::zero()];
        let expected = [Scalar::one(), Scalar::zero(), Scalar::zero()];
        mult_sc_vec(&mut intoo, &expected, 0);
        assert_eq!(intoo, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_point_mul() {
        assert_eq!(
            //TODO: check definition of one, what about curve One?
            Point::one(),
            Point {
                x: BaseField::one(),
                y: BaseField::zero(),
                z: BaseField::zero(),
            }
        );

        let mut inout = [Point::one(), Point::one(), Point::zero()];
        let scalars = [Scalar::one(), Scalar::zero(), Scalar::zero()];
        let expected = [
            Point::zero(),
            Point {
                x: BaseField::zero(),
                y: BaseField::one(),
                z: BaseField::zero(),
            },
            Point {
                x: BaseField::zero(),
                y: BaseField::one(),
                z: BaseField::zero(),
            },
        ];
        multp_vec(&mut inout, &scalars, 0);
        assert_eq!(inout, expected);
    }

    use std::fmt::Debug;

    fn assert_print<T: PartialEq + Debug>(ref_v: &Vec<T>, v_compar: Vec<T>) {
        let len = ref_v.len();
        let mut missing = len;
        let mut indexes = vec![];
        if *ref_v != v_compar {
            println!("!!!FAILED!!!");
            for i in 0..len {
                println!(
                    "***{}***\n{:08X?}\n***{}***\n{:08X?}\n***{}***",
                    i, ref_v[i], i, v_compar[i], i
                );
                for j in 0..len {
                    if ref_v[j] == v_compar[i] {
                        missing -= 1;
                        indexes.push(j);
                        break;
                    }
                }
            }
            assert_eq!(missing, 0, "there are {} missing els", missing);
            println!("swap indexes {:?}", indexes);
        }
    }

    fn get_limbs_csv_debug(filename: &str, limbsize: usize) -> Vec<u32> {
        let data_root_path = "../data/debug/";
        csv_to_u32_limbs(&format!("{}{}", data_root_path, filename), SCALAR_LIMBS)
    }

    fn get_scalar_limbs_csv_debug(filename: &str) -> Vec<u32> {
        get_limbs_csv_debug(filename, SCALAR_LIMBS)
    }
}
