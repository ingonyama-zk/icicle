use crate::{
    polynomial_ring::PolyRing,
    ring::{ScalarCfg, ScalarRing},
};
use icicle_core::{impl_jl_projection, impl_jl_projection_as_polyring};

impl_jl_projection!("labrador", ScalarRing, ScalarCfg);
impl_jl_projection_as_polyring!("labrador", PolyRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use crate::ring::ScalarRing;
    use icicle_core::{impl_jl_projection_polyring_tests, impl_jl_projection_tests};

    impl_jl_projection_tests!(ScalarRing);
    impl_jl_projection_polyring_tests!(PolyRing);
}
