use crate::field::Stark252Field;
use icicle_core::impl_sumcheck;

impl_sumcheck!("stark252", stark252, Stark252Field);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_sumcheck_tests;

    use crate::field::Stark252Field;

    impl_sumcheck_tests!(stark252, Stark252Field);
}
