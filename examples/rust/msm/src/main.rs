use icicle_bn254::curve::{CurveCfg, G1Projective, G1Affine, ScalarField};
use icicle_cuda_runtime::{memory::HostOrDeviceSlice, stream::CudaStream};
use icicle_core::{traits::FieldImpl, curve::Curve, msm};

#[cfg(feature = "compare")]
use halo2curves::{bn256, msm::best_multiexp, ff::PrimeField, group::Curve as halo2Curve};
#[cfg(feature = "compare")]
use ark_bn254::{Fq as arkFq, G1Affine as arkG1Affine};
#[cfg(feature = "compare")]
use ark_ff::{Field, biginteger::BigInteger256, bytes::FromBytes};

#[cfg(feature = "profile")]
use std::time::Instant;

use std::fs::File;
use std::io::BufReader;
use std::convert::TryInto;
use serde::de::DeserializeOwned;

/// Path to the directory where Arecibo data will be stored.
pub static ARECIBO_DATA: &str = ".arecibo_data";

/// Reads and deserializes data from a specified section and label.
fn read_arecibo_data<T: DeserializeOwned>(
    section: String,
    label: String,
) -> Vec<T> {
    let root_dir = home::home_dir().unwrap().join(ARECIBO_DATA);
    let section_path = root_dir.join(section);
    assert!(section_path.exists(), "Section directory does not exist");

    let file_path = section_path.join(label);
    assert!(file_path.exists(), "Data file does not exist");

    let file = File::open(file_path).expect("Failed to open data file");
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader).expect("Failed to read data")
}

fn icicle_to_bn256(point: &G1Affine) -> bn256::G1Affine {
    bn256::G1Affine { x: bn256::Fq::from_repr(point.x.to_bytes_le().try_into().unwrap()).unwrap(), y: bn256::Fq::from_repr(point.y.to_bytes_le().try_into().unwrap()).unwrap() }
}

fn icicle_to_ark(point: &G1Affine) -> arkG1Affine {
    arkG1Affine::new(arkFq::from_random_bytes(&point.x.to_bytes_le()).unwrap(), arkFq::from_random_bytes(&point.y.to_bytes_le()).unwrap(), false)
}

// cargo run --features=compare,profile -- --nocapture
fn main() {
    let section = "witness_0x02c29fabf43b87a73513f6ecbfb348c146809c1609c21b48333a8096700d63ad";
    let label_i = format!("len_8131411_{}", 0);
    // let section = "cross_term_0x02c29fabf43b87a73513f6ecbfb348c146809c1609c21b48333a8096700d63ad";
    // let label_i = format!("len_9873811_{}", 0);
    let scalars: Vec<[u64; 4]> = read_arecibo_data(section.to_string(), label_i.clone());
    let scalars: Vec<ScalarField> = scalars.iter().map(|limbs| ScalarField::from(*limbs)).collect();
    let size = scalars.len();
    let points = CurveCfg::generate_random_affine_points(size);

    // Setting Bn254 points and scalars
    let icicle_points = HostOrDeviceSlice::Host(points.clone());
    let icicle_scalars = HostOrDeviceSlice::Host(scalars.clone());

    let mut msm_results: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    let mut cfg = msm::MSMConfig::default();
    cfg.ctx
        .stream = &stream;
    cfg.is_async = true;

    #[cfg(feature = "profile")]
    let start = Instant::now();
    msm::msm(&icicle_scalars, &icicle_points, &cfg, &mut msm_results).unwrap();
    #[cfg(feature = "profile")]
    println!(
        "icicle GPU accelerated bn254 MSM took: {} ms",
        start
            .elapsed()
            .as_millis()
    );

    let mut msm_host_result = vec![G1Projective::zero(); 1];

    stream
        .synchronize()
        .unwrap();
    msm_results
        .copy_to_host(&mut msm_host_result[..])
        .unwrap();
    println!("MSM result: {:#?}", G1Affine::from(msm_host_result[0]));

    #[cfg(feature = "compare")]
    {
        let bn256_points: Vec<bn256::G1Affine> = points
            .iter()
            .map(|point| icicle_to_bn256(point))
            .collect();
        let bn256_witness: Vec<[u8; 32]> = read_arecibo_data(section.to_string(), label_i);
        let bn256_witness: Vec<bn256::Fr> = bn256_witness.iter().map(|limbs| bn256::Fr::from_repr(*limbs).unwrap()).collect();

        #[cfg(feature = "profile")]
        let start = Instant::now();
        let bn256_res = best_multiexp::<bn256::G1Affine>(&bn256_witness, &bn256_points);
        #[cfg(feature = "profile")]
        println!(
            "CPU version took: {} ms",
            start
                .elapsed()
                .as_millis()
        );

        let icicle_bn256_result = icicle_to_bn256(&G1Affine::from(msm_host_result[0]));
        println!(
            "bn254 MSM check vs. halo2curves is correct: {}",
            bn256_res.to_affine() == icicle_bn256_result
        );

        let ark_points: Vec<arkG1Affine> = points
            .iter()
            .map(|point| icicle_to_ark(point))
            .collect();
        let ark_scalars: Vec<BigInteger256> = scalars
            .iter()
            .map(|scalar| <BigInteger256 as FromBytes>::read(&scalar.to_bytes_le()[..]).unwrap())
            .collect();

        #[cfg(feature = "profile")]
        let start = Instant::now();
        let sppark_res = msm_cuda::multi_scalar_mult_arkworks(&ark_points, &ark_scalars);
        #[cfg(feature = "profile")]
        println!(
            "sppark GPU accelerated version took: {} ms",
            start
                .elapsed()
                .as_millis()
        );

        let icicle_ark_result = icicle_to_ark(&G1Affine::from(msm_host_result[0]));
        println!(
            "bn254 MSM check vs. sppark is correct: {}",
            sppark_res.eq(&icicle_ark_result)
        );
    }

    stream
        .destroy()
        .unwrap();
    println!("");
}
