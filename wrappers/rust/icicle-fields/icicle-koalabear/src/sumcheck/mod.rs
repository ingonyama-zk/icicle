use icicle_core::impl_sumcheck;

use crate::field::KoalabearField;

impl_sumcheck!("koalabear", koalabear, KoalabearField);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_sumcheck_tests;

    use crate::field::KoalabearField;

    impl_sumcheck_tests!(koalabear, KoalabearField);
}
