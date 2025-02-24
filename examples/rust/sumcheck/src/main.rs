pub mod transcript;
pub mod utils;

use icicle_bn254::curve::ScalarField as Fr;
use icicle_bn254::sumcheck::SumcheckWrapper;
use icicle_core::program::{PreDefinedProgram, ReturningValueProgram};
use icicle_core::sumcheck::{Sumcheck, SumcheckConfig, SumcheckTranscriptConfig};
use icicle_core::traits::FieldImpl;
use icicle_hash::blake3::Blake3;
use icicle_runtime::memory::HostSlice;
use merlin::Transcript;

use crate::transcript::*;
use crate::utils::*;
use clap::Parser;
use log::info;
use std::convert::TryInto;
use std::time::Instant;

const SAMPLES: usize = 1 << 22;

pub fn verify_proof(sumcheck: SumcheckWrapper, proof: icicle_bn254::sumcheck::SumcheckProof, claimed_sum: Fr) {
    let mut verifier_previous_transcript = Transcript::new(b"my_sumcheck");
    <Transcript as TranscriptProtocol<Fr>>::append_data(&mut verifier_previous_transcript, b"public", &claimed_sum);
    //get seed based on previous state
    let verifier_seed_rng =
        <Transcript as TranscriptProtocol<Fr>>::challenge_scalar(&mut verifier_previous_transcript, b"seeded");

    //define verifier FS config
    let leaf_size = (Fr::one())
        .to_bytes_le()
        .len()
        .try_into()
        .unwrap();
    let hasher = Blake3::new(leaf_size).unwrap();
    let verifier_transcript_config = SumcheckTranscriptConfig::new(
        &hasher,
        b"start_sumcheck".to_vec(),
        b"round_poly".to_vec(),
        b"round_challenge".to_vec(),
        true,
        verifier_seed_rng,
    );

    let sumcheck = <icicle_bn254::sumcheck::SumcheckWrapper as Sumcheck>::new().unwrap();
    let proof_validty = sumcheck.verify(&proof, claimed_sum, &verifier_transcript_config);

    match proof_validty {
        Ok(true) => println!("Valid proof!"), // Verification succeeded
        Ok(false) => {
            eprintln!("Sumcheck proof not valid");
            assert!(false, "Sumcheck proof verification failed!");
        }
        Err(err) => {
            eprintln!("Error in verification {:?}",err);
            assert!(false, "Sumcheck proof verification encountered an error!");
        }
    }
}

//RUST_LOG=info cargo run --release --package sumcheck --bin sumcheck
pub fn main() {
    env_logger::init();

    let args = Args::parse();
    try_load_and_set_backend_device(&args);
    //simulate previous state
    let mut prover_previous_transcript = Transcript::new(b"my_sumcheck");

    let gen_data_time = Instant::now();
    let poly_a = generate_random_vector::<Fr>(SAMPLES);
    let poly_b = generate_random_vector::<Fr>(SAMPLES);
    let poly_c = generate_random_vector::<Fr>(SAMPLES);
    let poly_e = generate_random_vector::<Fr>(SAMPLES);

    info!(
        "Generate e,A,B,C of log size {:?}, time {:?}",
        SAMPLES.ilog2(),
        gen_data_time.elapsed()
    );
    let compute_sum_time = Instant::now();
    //compute claimed sum
    let temp: Vec<Fr> = poly_a
        .iter()
        .zip(poly_b.iter())
        .zip(poly_c.iter())
        .zip(poly_e.iter())
        .map(|(((a, b), c), e)| *a * *b * *e - *c * *e)
        .collect();
    let claimed_sum = temp
        .iter()
        .fold(Fr::zero(), |acc, &a| acc + a);
    info!("Compute claimed sum time {:?}", compute_sum_time.elapsed());

    //add claimed sum to transcript to simulate previous state
    <Transcript as TranscriptProtocol<Fr>>::append_data(&mut prover_previous_transcript, b"public", &claimed_sum);
    //get seed based on previous state
    let seed_rng = <Transcript as TranscriptProtocol<Fr>>::challenge_scalar(&mut prover_previous_transcript, b"seeded");

    let leaf_size = (Fr::one())
        .to_bytes_le()
        .len()
        .try_into()
        .unwrap();
    let hasher = Blake3::new(leaf_size).unwrap();

    //define sumcheck config
    let sumcheck_config = SumcheckConfig::default();

    let mle_poly_hosts = vec![
        HostSlice::<Fr>::from_slice(&poly_a),
        HostSlice::<Fr>::from_slice(&poly_b),
        HostSlice::<Fr>::from_slice(&poly_c),
        HostSlice::<Fr>::from_slice(&poly_e),
    ];
    let sumcheck = <icicle_bn254::sumcheck::SumcheckWrapper as Sumcheck>::new().unwrap();

    let transcript_config = SumcheckTranscriptConfig::new(
        &hasher,
        b"start_sumcheck".to_vec(),
        b"round_poly".to_vec(),
        b"round_challenge".to_vec(),
        true,
        seed_rng,
    );
    //try different combine functions!
    let combine_function =
        <icicle_bn254::program::FieldReturningValueProgram as ReturningValueProgram>::new_predefined(
            PreDefinedProgram::EQtimesABminusC,
        )
        .unwrap();
    let prover_time = Instant::now();
    let proof = sumcheck.prove(
        &mle_poly_hosts,
        SAMPLES
            .try_into()
            .unwrap(),
        claimed_sum,
        combine_function,
        &transcript_config,
        &sumcheck_config,
    );
    info!("Prover time {:?}", prover_time.elapsed());
    set_backend_cpu();
    //experiment with fake sum or fake proof
    let verify_time = Instant::now();
    verify_proof(sumcheck, proof, claimed_sum);
    info!("verify time {:?}", verify_time.elapsed());
    info!("total time {:?}", gen_data_time.elapsed());
}
