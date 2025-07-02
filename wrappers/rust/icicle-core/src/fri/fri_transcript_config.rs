use crate::hash::Hasher;
use crate::hash::HasherHandle;
use crate::field::PrimeField;

const DEFAULT_DOMAIN_SEPARATOR_LABEL: &str = "domain_separator_label";
const DEFAULT_ROUND_CHALLENGE_LABEL: &str = "round_challenge_label";
const DEFAULT_COMMIT_PHASE_LABEL: &str = "commit_phase_label";
const DEFAULT_NONCE_LABEL: &str = "nonce_label";
const DEFAULT_PUBLIC_STATE: &[u8] = b"";

/// Configuration for encoding and hashing messages in the FRI protocol.
pub struct FriTranscriptConfig<'a, F: PrimeField + 'a> {
    pub hash: &'a Hasher,
    pub domain_separator_label: String,
    pub round_challenge_label: String,
    pub commit_phase_label: String,
    pub nonce_label: String,
    pub public_state: Vec<u8>,
    pub seed_rng: F,
}

impl<'a, F: PrimeField + 'a> FriTranscriptConfig<'a, F> {
    /// Creates a new `FriTranscriptConfig` with custom labels.
    pub fn new(
        hash: &'a Hasher,
        domain_separator_label: &str,
        round_challenge_label: &str,
        commit_phase_label: &str,
        nonce_label: &str,
        public_state: &[u8],
        seed_rng: F,
    ) -> Self {
        Self {
            hash,
            domain_separator_label: domain_separator_label.to_string(),
            round_challenge_label: round_challenge_label.to_string(),
            commit_phase_label: commit_phase_label.to_string(),
            nonce_label: nonce_label.to_string(),
            public_state: public_state.to_vec(),
            seed_rng,
        }
    }

    /// Creates a `FriTranscriptConfig` with default labels.
    pub fn new_default_labels(hash: &'a Hasher, seed_rng: F) -> Self {
        Self::new(
            hash,
            DEFAULT_DOMAIN_SEPARATOR_LABEL,
            DEFAULT_ROUND_CHALLENGE_LABEL,
            DEFAULT_COMMIT_PHASE_LABEL,
            DEFAULT_NONCE_LABEL,
            DEFAULT_PUBLIC_STATE,
            seed_rng,
        )
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct FFIFriTranscriptConfig<F: PrimeField> {
    hasher_handle: HasherHandle,
    domain_separator_label: *const u8,
    domain_separator_label_len: usize,
    round_challenge_label: *const u8,
    round_challenge_label_len: usize,
    commit_phase_label: *const u8,
    commit_phase_label_len: usize,
    nonce_label: *const u8,
    nonce_label_label: usize,
    public_state: *const u8,
    public_state_len: usize,
    seed_rng: *const F,
}

impl<'a, F: PrimeField> From<&FriTranscriptConfig<'a, F>> for FFIFriTranscriptConfig<F> {
    fn from(config: &FriTranscriptConfig<'a, F>) -> Self {
        Self {
            hasher_handle: config
                .hash
                .handle,
            domain_separator_label: config
                .domain_separator_label
                .as_ptr(),
            domain_separator_label_len: config
                .domain_separator_label
                .len(),
            round_challenge_label: config
                .round_challenge_label
                .as_ptr(),
            round_challenge_label_len: config
                .round_challenge_label
                .len(),
            commit_phase_label: config
                .commit_phase_label
                .as_ptr(),
            commit_phase_label_len: config
                .commit_phase_label
                .len(),
            nonce_label: config
                .nonce_label
                .as_ptr(),
            nonce_label_label: config
                .nonce_label
                .len(),
            public_state: config
                .public_state
                .as_ptr(),
            public_state_len: config
                .public_state
                .len(),
            seed_rng: &config.seed_rng,
        }
    }
}
