use crate::bignum::BigNum;
use crate::hash::Hasher;
use crate::program::{PreDefinedProgram, ReturningValueProgramImpl};
use crate::ring::IntegerRing;
use crate::sumcheck::{Sumcheck, SumcheckConfig, SumcheckProofOps, SumcheckTranscriptConfig};
use crate::traits::GenerateRandom;
use icicle_runtime::memory::{DeviceSlice, DeviceVec, HostSlice, IntoIcicleSlice};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Tests the `SumcheckTranscriptConfig` struct with different constructors.
pub fn check_sumcheck_transcript_config<F: IntegerRing>(hash: &Hasher)
where
    F: GenerateRandom,
{
    // Generate a random seed for the test.
    let seed_rng = F::generate_random(1)[0];

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

/// Tests the basic functionality of the Sumcheck protocol on CPU.
///
/// This test verifies the core proving and verification functionality:
/// 1. Creates random MLE polynomials
/// 2. Computes a claimed sum using the EQtimesABminusC predefined program
/// 3. Generates a proof on CPU
/// 4. Verifies the proof
///
/// The test uses a polynomial size of 2^13 and 4 MLE polynomials.
pub fn check_sumcheck_simple<SW, P>(hash: &Hasher)
where
    SW: Sumcheck,
    SW::Field: GenerateRandom,
    P: ReturningValueProgramImpl,
{
    let log_mle_poly_size = 13u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let nof_mle_poly = 4;
    // Generate a random seed for the test.
    let seed_rng = SW::Field::generate_random(1)[0];

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
        let mle_poly_random = SW::Field::generate_random(mle_poly_size);
        mle_polys.push(mle_poly_random);
    }

    let mut claimed_sum = SW::Field::zero();
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
        .map(|poly| poly.into_slice())
        .collect::<Vec<&HostSlice<<SW as Sumcheck>::Field>>>();

    let sumcheck = SW::new().unwrap();
    let combine_func = P::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();
    let sumcheck_config = SumcheckConfig::default();
    // Generate a proof using the `prove` method.
    let proof = sumcheck
        .prove(
            mle_poly_hosts.as_slice(),
            mle_poly_size as u64,
            claimed_sum,
            combine_func,
            &config,
            &sumcheck_config,
        )
        .unwrap();
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

/// Tests the Sumcheck protocol on GPU/device.
///
/// Similar to `check_sumcheck_simple` but verifies the protocol works correctly
/// when polynomials are stored on the device. The test:
/// 1. Creates random MLE polynomials
/// 2. Copies them to device memory
/// 3. Generates a proof using device data
/// 4. Verifies the proof
pub fn check_sumcheck_simple_device<SW, P>(hash: &Hasher)
where
    SW: Sumcheck,
    SW::Field: GenerateRandom,
    P: ReturningValueProgramImpl,
{
    let log_mle_poly_size = 13u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let nof_mle_poly = 4;

    let seed_rng = SW::Field::generate_random(1)[0];

    let mut mle_polys = Vec::with_capacity(nof_mle_poly);
    for _ in 0..nof_mle_poly {
        let mle_poly_random = SW::Field::generate_random(mle_poly_size);
        mle_polys.push(mle_poly_random);
    }

    let mut claimed_sum = SW::Field::zero();
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
        .map(|poly| poly.into_slice())
        .collect::<Vec<&HostSlice<<SW as Sumcheck>::Field>>>();

    let mut device_mle_polys = Vec::with_capacity(nof_mle_poly);
    for i in 0..nof_mle_poly {
        let mut device_slice = DeviceVec::malloc(mle_poly_size);
        device_slice
            .copy_from_host(mle_poly_hosts[i])
            .unwrap();
        device_mle_polys.push(device_slice);
    }

    let mle_polys_device: Vec<&DeviceSlice<<SW as Sumcheck>::Field>> = device_mle_polys
        .iter()
        .map(|s| s.into_slice())
        .collect();
    let device_mle_polys_slice = mle_polys_device.as_slice();

    let sumcheck = SW::new().unwrap();
    let combine_func = P::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();
    let sumcheck_config = SumcheckConfig::default();
    // Generate a proof using the `prove` method.
    let proof = sumcheck
        .prove(
            device_mle_polys_slice,
            mle_poly_size as u64,
            claimed_sum,
            combine_func,
            &config,
            &sumcheck_config,
        )
        .unwrap();
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

/// Tests the Sumcheck protocol with a user-defined combining function.
///
/// This test verifies that the protocol works correctly with custom polynomial
/// combination functions. It:
/// 1. Creates random MLE polynomials
/// 2. Defines a custom combining function: a * b + d - 2 * c
/// 3. Generates a proof using the custom function
/// 4. Verifies the proof
pub fn check_sumcheck_user_defined_combine<SW, P>(hash: &Hasher)
where
    SW: Sumcheck,
    SW::Field: GenerateRandom,
    P: ReturningValueProgramImpl,
{
    let log_mle_poly_size = 13u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let nof_mle_poly = 4;
    // Generate a random seed for the test.
    let seed_rng = SW::Field::generate_random(1)[0];

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
        let mle_poly_random = SW::Field::generate_random(mle_poly_size);
        mle_polys.push(mle_poly_random);
    }

    let mut claimed_sum = SW::Field::zero();
    for i in 0..mle_poly_size {
        let a = mle_polys[0][i];
        let b = mle_polys[1][i];
        let c = mle_polys[2][i];
        let d = mle_polys[3][i];
        claimed_sum = claimed_sum + a * b - c * SW::Field::from(2) + d;
    }

    let user_combine = |vars: &mut Vec<P::ProgSymbol>| -> P::ProgSymbol {
        let a = vars[0]; // Shallow copies pointing to the same memory in the backend
        let b = vars[1];
        let c = vars[2];
        let d = vars[3];
        a * b + d - c * P::Ring::from(2)
    };

    /****** Begin CPU Proof ******/
    let mle_poly_hosts = mle_polys
        .iter()
        .map(|poly| poly.into_slice())
        .collect::<Vec<&HostSlice<<SW as Sumcheck>::Field>>>();

    let sumcheck = SW::new().unwrap();
    let combine_func = P::new(user_combine, /* nof_parameters = */ 4).unwrap();
    let sumcheck_config = SumcheckConfig::default();
    // Generate a proof using the `prove` method.
    let proof = sumcheck
        .prove(
            mle_poly_hosts.as_slice(),
            mle_poly_size as u64,
            claimed_sum,
            combine_func,
            &config,
            &sumcheck_config,
        )
        .unwrap();
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

/// Tests the serialization and deserialization of Sumcheck proofs.
///
/// This test verifies that proofs can be correctly serialized and deserialized:
/// 1. Generates a proof using a custom combining function
/// 2. Serializes the proof using the provided serializer
/// 3. Deserializes the proof using the provided deserializer
/// 4. Verifies that the round polynomials match the original proof
pub fn check_sumcheck_proof_serialization<SW, P, S, D, T>(hash: &Hasher, serialize: S, deserialize: D)
where
    SW: Sumcheck,
    SW::Field: GenerateRandom,
    P: ReturningValueProgramImpl,
    SW::Proof: Serialize + DeserializeOwned,
    S: Fn(&SW::Proof) -> T,
    D: Fn(&T) -> SW::Proof,
{
    let log_mle_poly_size = 13u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let nof_mle_poly = 4;
    // Generate a random seed for the test.
    let seed_rng = SW::Field::generate_random(1)[0];

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
        let mle_poly_random = SW::Field::generate_random(mle_poly_size);
        mle_polys.push(mle_poly_random);
    }

    let mut claimed_sum = SW::Field::zero();
    for i in 0..mle_poly_size {
        let a = mle_polys[0][i];
        let b = mle_polys[1][i];
        let c = mle_polys[2][i];
        let d = mle_polys[3][i];
        claimed_sum = claimed_sum + a * b - c * SW::Field::from(2) + d;
    }

    let user_combine = |vars: &mut Vec<P::ProgSymbol>| -> P::ProgSymbol {
        let a = vars[0]; // Shallow copies pointing to the same memory in the backend
        let b = vars[1];
        let c = vars[2];
        let d = vars[3];
        a * b + d - c * P::Ring::from(2)
    };

    /****** Begin CPU Proof ******/
    let mle_poly_hosts = mle_polys
        .iter()
        .map(|poly| poly.into_slice())
        .collect::<Vec<&HostSlice<<SW as Sumcheck>::Field>>>();

    let sumcheck = SW::new().unwrap();
    let combine_func = P::new(user_combine, /* nof_parameters = */ 4).unwrap();
    let sumcheck_config = SumcheckConfig::default();
    // Generate a proof using the `prove` method.
    let proof = sumcheck
        .prove(
            mle_poly_hosts.as_slice(),
            mle_poly_size as u64,
            claimed_sum,
            combine_func,
            &config,
            &sumcheck_config,
        )
        .unwrap();
    /****** End CPU Proof ******/

    let proof_round_polys =
        <<SW as Sumcheck>::Proof as SumcheckProofOps<<SW as Sumcheck>::Field>>::get_round_polys(&proof).unwrap();

    let proof_as_sumcheck_proof: <SW as Sumcheck>::Proof = <SW as Sumcheck>::Proof::from(proof_round_polys);

    // === Serialize ===
    let serialized_proof = serialize(&proof_as_sumcheck_proof);
    let deserialized_proof: SW::Proof = deserialize(&serialized_proof);

    let round_polys_original = proof_as_sumcheck_proof
        .get_round_polys()
        .unwrap();
    let round_polys_deserialized = deserialized_proof
        .get_round_polys()
        .unwrap();

    assert_eq!(round_polys_original, round_polys_deserialized);
}

/// Tests the challenge vector functionality of the Sumcheck protocol.
///
/// This test verifies the properties of the challenge vector:
/// 1. Initial state is empty
/// 2. After proof generation:
///    - Length equals log2(mle_poly_size)
///    - First challenge is zero (protocol requirement)
///    - All other challenges are non-zero
///
/// The test uses a small polynomial size (2^4) for efficiency and three
/// MLE polynomials (A, B, C) with the ABminusC predefined program.
pub fn check_sumcheck_challenge_vector<SW, P>(hash: &Hasher)
where
    SW: Sumcheck,
    SW::Field: GenerateRandom,
    P: ReturningValueProgramImpl,
{
    // Create a simple sumcheck instance
    let sumcheck = SW::new().unwrap();

    // Get the initial challenge vector and verify it's empty
    let challenge_vector = sumcheck
        .get_challenge_vector()
        .unwrap();
    assert!(
        challenge_vector.is_empty(),
        "Challenge vector should be empty before proving"
    );

    // Run a simple sumcheck proof to populate the challenge vector
    let log_mle_poly_size = 4u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let seed_rng = SW::Field::generate_random(1)[0];

    let config = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true,
        seed_rng,
    );

    // Generate three polynomials for A, B, and C
    let mle_poly_a = SW::Field::generate_random(mle_poly_size);
    let mle_poly_b = SW::Field::generate_random(mle_poly_size);
    let mle_poly_c = SW::Field::generate_random(mle_poly_size);

    // Ensure the polynomials are not empty
    assert!(!mle_poly_a.is_empty(), "MLE polynomial A should not be empty");
    assert!(!mle_poly_b.is_empty(), "MLE polynomial B should not be empty");
    assert!(!mle_poly_c.is_empty(), "MLE polynomial C should not be empty");

    let mle_poly_a_host = mle_poly_a.into_slice();
    let mle_poly_b_host = mle_poly_b.into_slice();
    let mle_poly_c_host = mle_poly_c.into_slice();

    // Create a vector of references that will live for the duration of the prove call
    let mle_poly_refs: Vec<&HostSlice<_>> = vec![&mle_poly_a_host, &mle_poly_b_host, &mle_poly_c_host];

    // Calculate claimed sum: sum(A * B - C)
    let claimed_sum = mle_poly_a
        .iter()
        .zip(mle_poly_b.iter())
        .zip(mle_poly_c.iter())
        .fold(<SW as Sumcheck>::Field::zero(), |acc, ((&a, &b), &c)| acc + (a * b - c));

    let combine_func = P::new_predefined(PreDefinedProgram::ABminusC).unwrap();
    let sumcheck_config = SumcheckConfig::default();

    // Generate proof
    let _proof = {
        let mle_poly_refs = mle_poly_refs.as_slice();
        sumcheck.prove(
            mle_poly_refs,
            mle_poly_size as u64,
            claimed_sum,
            combine_func,
            &config,
            &sumcheck_config,
        )
    };

    // Get the challenge vector after proving
    let challenge_vector = sumcheck
        .get_challenge_vector()
        .unwrap();

    // Verify challenge vector properties
    assert_eq!(
        challenge_vector.len(),
        log_mle_poly_size as usize,
        "Challenge vector should have length equal to log2(mle_poly_size)"
    );

    // First challenge should be zero (as per sumcheck protocol)
    assert_eq!(
        challenge_vector[0],
        <SW as Sumcheck>::Field::zero(),
        "First challenge should be zero"
    );

    // All other challenges should be non-zero
    for (i, &challenge) in challenge_vector
        .iter()
        .enumerate()
        .skip(1)
    {
        assert!(
            challenge != <SW as Sumcheck>::Field::zero(),
            "Challenge at index {} should be non-zero",
            i
        );
    }
}
