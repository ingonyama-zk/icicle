use icicle_core::field::Field;
use icicle_core::traits::GenerateRandom;
use merlin::Transcript;

pub trait TranscriptProtocol<F: Field> {
    fn challenge_scalar(&mut self, label: &'static [u8]) -> F;
    /// Append a `scalar` with the given `label`.
    fn append_data(&mut self, label: &'static [u8], scalar: &F);
}

impl<F: Field> TranscriptProtocol<F> for Transcript
where
    F: GenerateRandom,
{
    fn challenge_scalar(&mut self, label: &'static [u8]) -> F {
        let mut buf = [0u8; 64];
        self.challenge_bytes(label, &mut buf);
        F::from_bytes_le(&buf)
    }
    fn append_data(&mut self, label: &'static [u8], scalar: &F) {
        self.append_message(label, &scalar.to_bytes_le());
    }
}
