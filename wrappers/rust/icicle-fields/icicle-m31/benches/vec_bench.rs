use icicle_m31::field::{QuarticExtensionField, ScalarField};

use icicle_core::impl_vec_ops_bench;

mod m31 {
    use super::*;
    impl_vec_ops_bench!("m31", ScalarField);
}
mod m31_q_extension {
    use super::*;
    impl_vec_ops_bench!("m31_q_extension", QuarticExtensionField);
}

use criterion::criterion_main;

criterion_main!(m31::benches, m31_q_extension::benches);
