use crate::hash::Hasher;
use crate::program::{PreDefinedProgram, ReturningValueProgram};
use crate::sumcheck::{Sumcheck, SumcheckConfig, SumcheckProofOps, SumcheckTranscriptConfig};
use crate::traits::{FieldImpl, GenerateRandom};
use icicle_runtime::memory::{DeviceSlice, DeviceVec, HostSlice};

/// Tests the `SumcheckTranscriptConfig` struct with different constructors.
pub fn check_sumcheck_transcript_config<F: FieldImpl>(hash: &Hasher)
where
    <F as FieldImpl>::Config: GenerateRandom<F>,
{
    // Generate a random seed for the test.
    let seed_rng = F::Config::generate_random(1)[0];

    // Test `new` constructor
    let config1 = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true, // little endian
        seed_rng,
    );

    // Verify that the fields are correctly initialized
    assert_eq!(config1.domain_separator_label, b"DomainLabel");
    assert_eq!(config1.round_poly_label, b"PolyLabel");
    assert_eq!(config1.round_challenge_label, b"ChallengeLabel");
    assert!(config1.little_endian);
    assert_eq!(config1.seed_rng, seed_rng);

    // Test `from_string_labels` constructor
    let config2 = SumcheckTranscriptConfig::from_string_labels(
        hash,
        "DomainLabel",
        "PolyLabel",
        "ChallengeLabel",
        false, // big endian
        seed_rng,
    );

    // Verify that the fields are correctly initialized
    assert_eq!(config2.domain_separator_label, b"DomainLabel");
    assert_eq!(config2.round_poly_label, b"PolyLabel");
    assert_eq!(config2.round_challenge_label, b"ChallengeLabel");
    assert!(!config2.little_endian);
    assert_eq!(config2.seed_rng, seed_rng);
}

/// Tests the `Sumcheck` struct's basic functionality, including proving and verifying.
pub fn check_sumcheck_simple<SW, P>(hash: &Hasher)
where
    SW: Sumcheck,
    P: ReturningValueProgram,
{
    let log_mle_poly_size = 13u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let nof_mle_poly = 4;
    // Generate a random seed for the test.
    let seed_rng = <<SW as Sumcheck>::FieldConfig>::generate_random(1)[0];

    // Create a transcript configuration.
    let config = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true, // little endian
        seed_rng,
    );

    let mut mle_polys = Vec::with_capacity(nof_mle_poly);
    for _ in 0..nof_mle_poly {
        let mle_poly_random = <<SW as Sumcheck>::FieldConfig>::generate_random(mle_poly_size);
        mle_polys.push(mle_poly_random);
    }

    let mut claimed_sum = <<SW as Sumcheck>::Field as FieldImpl>::zero();
    for i in 0..mle_poly_size {
        let a = mle_polys[0][i];
        let b = mle_polys[1][i];
        let c = mle_polys[2][i];
        let eq = mle_polys[3][i];
        claimed_sum = claimed_sum + (a * b - c) * eq;
    }

    /****** Begin CPU Proof ******/
    let mle_poly_hosts = mle_polys
        .iter()
        .map(|poly| HostSlice::from_slice(poly))
        .collect::<Vec<&HostSlice<<SW as Sumcheck>::Field>>>();

    let sumcheck = SW::new().unwrap();
    let combine_func = P::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();
    let sumcheck_config = SumcheckConfig::default();
    // Generate a proof using the `prove` method.
    let proof = sumcheck.prove(
        mle_poly_hosts.as_slice(),
        mle_poly_size as u64,
        claimed_sum,
        combine_func,
        &config,
        &sumcheck_config,
    );
    /****** End CPU Proof ******/

    /****** Obtain Proof Round Polys ******/
    let proof_round_polys =
        <<SW as Sumcheck>::Proof as SumcheckProofOps<<SW as Sumcheck>::Field>>::get_round_polys(&proof).unwrap();

    /********** Verifier deserializes proof data *********/
    let proof_as_sumcheck_proof: <SW as Sumcheck>::Proof = <SW as Sumcheck>::Proof::from(proof_round_polys);

    // Verify the proof.
    let valid = sumcheck
        .verify(&proof_as_sumcheck_proof, claimed_sum, &config)
        .unwrap();

    assert!(valid);
}

pub fn check_sumcheck_simple_device<SW, P>(hash: &Hasher)
where
    SW: Sumcheck,
    P: ReturningValueProgram,
{
    let log_mle_poly_size = 13u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let nof_mle_poly = 4;

    let seed_rng = <<SW as Sumcheck>::FieldConfig>::generate_random(1)[0];

    let mut mle_polys = Vec::with_capacity(nof_mle_poly);
    for _ in 0..nof_mle_poly {
        let mle_poly_random = <<SW as Sumcheck>::FieldConfig>::generate_random(mle_poly_size);
        mle_polys.push(mle_poly_random);
    }

    let mut claimed_sum = <<SW as Sumcheck>::Field as FieldImpl>::zero();
    for i in 0..mle_poly_size {
        let a = mle_polys[0][i];
        let b = mle_polys[1][i];
        let c = mle_polys[2][i];
        let eq = mle_polys[3][i];
        claimed_sum = claimed_sum + (a * b - c) * eq;
    }

    /****** Begin Device Proof ******/
    let config = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true, // little endian
        seed_rng,
    );

    let mle_poly_hosts = mle_polys
        .iter()
        .map(|poly| HostSlice::from_slice(poly))
        .collect::<Vec<&HostSlice<<SW as Sumcheck>::Field>>>();

    let mut device_mle_polys = Vec::with_capacity(nof_mle_poly);
    for i in 0..nof_mle_poly {
        let mut device_slice = DeviceVec::device_malloc(mle_poly_size).unwrap();
        device_slice
            .copy_from_host(mle_poly_hosts[i])
            .unwrap();
        device_mle_polys.push(device_slice);
    }

    let mle_polys_device: Vec<&DeviceSlice<<SW as Sumcheck>::Field>> = device_mle_polys
        .iter()
        .map(|s| &s[..])
        .collect();
    let device_mle_polys_slice = mle_polys_device.as_slice();

    let sumcheck = SW::new().unwrap();
    let combine_func = P::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();
    let sumcheck_config = SumcheckConfig::default();
    // Generate a proof using the `prove` method.
    let proof = sumcheck.prove(
        device_mle_polys_slice,
        mle_poly_size as u64,
        claimed_sum,
        combine_func,
        &config,
        &sumcheck_config,
    );
    /****** End Device Proof ******/

    /****** Obtain Proof Round Polys ******/
    let proof_round_polys =
        <<SW as Sumcheck>::Proof as SumcheckProofOps<<SW as Sumcheck>::Field>>::get_round_polys(&proof).unwrap();

    /********** Verifier deserializes proof data *********/
    let proof_as_sumcheck_proof: <SW as Sumcheck>::Proof = <SW as Sumcheck>::Proof::from(proof_round_polys);

    // Verify the proof.
    let valid = sumcheck
        .verify(&proof_as_sumcheck_proof, claimed_sum, &config)
        .unwrap();

    assert!(valid);
}

pub fn check_sumcheck_user_defined_combine<SW, P>(hash: &Hasher)
where
    SW: Sumcheck,
    P: ReturningValueProgram,
{
    let log_mle_poly_size = 13u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let nof_mle_poly = 4;
    // Generate a random seed for the test.
    let seed_rng = <<SW as Sumcheck>::FieldConfig>::generate_random(1)[0];

    // Create a transcript configuration.
    let config = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true, // little endian
        seed_rng,
    );

    let mut mle_polys = Vec::with_capacity(nof_mle_poly);
    for _ in 0..nof_mle_poly {
        let mle_poly_random = <<SW as Sumcheck>::FieldConfig>::generate_random(mle_poly_size);
        mle_polys.push(mle_poly_random);
    }

    let mut claimed_sum = <<SW as Sumcheck>::Field as FieldImpl>::zero();
    for i in 0..mle_poly_size {
        let a = mle_polys[0][i];
        let b = mle_polys[1][i];
        let c = mle_polys[2][i];
        let d = mle_polys[3][i];
        claimed_sum = claimed_sum + a * b - c * SW::Field::from_u32(2) + d;
    }

    let user_combine = |vars: &mut Vec<P::ProgSymbol>| -> P::ProgSymbol {
        let a = vars[0]; // Shallow copies pointing to the same memory in the backend
        let b = vars[1];
        let c = vars[2];
        let d = vars[3];
        return a * b + d - c * P::Field::from_u32(2);
    };

    /****** Begin CPU Proof ******/
    let mle_poly_hosts = mle_polys
        .iter()
        .map(|poly| HostSlice::from_slice(poly))
        .collect::<Vec<&HostSlice<<SW as Sumcheck>::Field>>>();

    let sumcheck = SW::new().unwrap();
    let combine_func = P::new(user_combine, /* nof_parameters = */ 4).unwrap();
    let sumcheck_config = SumcheckConfig::default();
    // Generate a proof using the `prove` method.
    let proof = sumcheck.prove(
        mle_poly_hosts.as_slice(),
        mle_poly_size as u64,
        claimed_sum,
        combine_func,
        &config,
        &sumcheck_config,
    );
    /****** End CPU Proof ******/

    /****** Obtain Proof Round Polys ******/
    let proof_round_polys =
        <<SW as Sumcheck>::Proof as SumcheckProofOps<<SW as Sumcheck>::Field>>::get_round_polys(&proof).unwrap();

    /********** Verifier deserializes proof data *********/
    let proof_as_sumcheck_proof: <SW as Sumcheck>::Proof = <SW as Sumcheck>::Proof::from(proof_round_polys);

    // Verify the proof.
    let valid = sumcheck
        .verify(&proof_as_sumcheck_proof, claimed_sum, &config)
        .unwrap();

    assert!(valid);
}
