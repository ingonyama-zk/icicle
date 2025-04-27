use crate::field::BabybearField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("babybear", babybear, BabybearField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::BabybearField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(babybear, BabybearField);
}
