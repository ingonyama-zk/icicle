use icicle_m31::field::{ExtensionField, ScalarField};

use icicle_core::impl_vec_ops_bench;

mod m31 {
    use super::*;
    impl_vec_ops_bench!("m31", ScalarField);
}
mod m31_extension {
    use super::*;
    impl_vec_ops_bench!("m31_extension", ExtensionField);
}

use criterion::criterion_main;

criterion_main!(m31::benches, m31_extension::benches);
