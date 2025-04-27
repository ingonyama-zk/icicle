use icicle_core::impl_sumcheck;

use crate::field::M31Field;

impl_sumcheck!("m31", m31, M31Field);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_sumcheck_tests;

    use crate::field::M31Field;

    impl_sumcheck_tests!(m31, M31Field);
}
