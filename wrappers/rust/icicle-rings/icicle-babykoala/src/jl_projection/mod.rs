use crate::{polynomial_ring::PolyRing, ring::ScalarRing};
use icicle_core::impl_jl_projection;
use icicle_core::impl_jl_projection_as_polyring;

impl_jl_projection!("babykoala", ScalarRing);
impl_jl_projection_as_polyring!("babykoala", PolyRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use crate::ring::ScalarRing;
    use icicle_core::{impl_jl_projection_polyring_tests, impl_jl_projection_tests};

    impl_jl_projection_tests!(ScalarRing);
    impl_jl_projection_polyring_tests!(PolyRing);
}
