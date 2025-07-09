# Pairings

The pairing interface in `icicle-core` gives you access to efficient bilinear pairings on supported curves (BN254, BLS12-377, BLS12-381, BW6-761, …).  Everything is wrapped in a safe, idiomatic Rust layer that automatically moves data between host and device memory when necessary.

---
## Trait & helper function

```rust
use icicle_core::{field::Field, projective::Projective};
use icicle_runtime::errors::IcicleError;

pub trait Pairing<P1: Projective, P2: Projective, F: Field> {
    fn pairing(p: &P1::Affine, q: &P2::Affine) -> Result<F, IcicleError>;
}

/// Convenience free-function that dispatches to the trait implementation
pub fn pairing<P1, P2, F>(p: &P1::Affine, q: &P2::Affine) -> Result<F, IcicleError>
where
    P1: Projective + Pairing<P1, P2, F>,
    P2: Projective,
    F: Field,
{
    P1::pairing(p, q)
}
```

Each curve crate that supports pairings provides the concrete implementation via the `impl_pairing!` macro.  For BN254 (which is equipped with a single‐thread pairing on G1×G2) the types look like:

```rust
use icicle_bn254::curve::{G1Projective, G2Projective};
use icicle_bn254::pairing::PairingTargetField; // 12-degree extension field Fₚ¹²
```

---
## Example

```rust
use icicle_bn254::curve::{G1Projective, G2Projective};
use icicle_core::pairing::pairing; // helper free-function
use icicle_core::traits::GenerateRandom;

// Generate random points (G1 + G2 live in different groups)
let p = G1Projective::generate_random(1)[0].to_affine();
let q = G2Projective::generate_random(1)[0].to_affine();
// Compute the pairing e(P, Q)
let gt = pairing::<G1Projective, G2Projective, _>(&p, &q).unwrap();
println!("e(P, Q) = {:?}", gt);
```

The return value `gt` lives in the **target field** (e.g. Fₚ¹² for BN254/BLS12 curves).  The exact name of the type depends on the curve crate – consult the module `pairing` inside each crate.
