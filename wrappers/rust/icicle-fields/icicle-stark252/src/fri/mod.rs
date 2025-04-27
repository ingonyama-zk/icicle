use icicle_core::impl_fri;

use crate::field::Stark252Field;

impl_fri!("stark252", stark252_fri, Stark252Field);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::field::Stark252Field;

    impl_fri_tests!(stark252_fri_test, Stark252Field, Stark252Field);
}
