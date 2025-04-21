#pragma once

#include "icicle/fields/cubic_extension.h"
#include "icicle/curves/params/bls12_381.h"
#include "icicle/pairing/models/bls12.h"

namespace pairing_bls12_381 {
  using namespace bls12_381;
  using namespace icicle_bls12_pairing;

  struct PairingConfig {
    static constexpr storage<2> Z = {0x00010000, 0xd2010000};
    static constexpr bool Z_IS_NEGATIVE = true;
    static constexpr int Z_NAF[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,
                                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, -1, 0, 1};

    static constexpr TwistType TWIST_TYPE = TwistType::M;

    static constexpr g2_point_field_t CUBIC_NONRESIDUE = fq6_config::nonresidue;

    static void mul_fp2_by_nonresidue(g2_point_field_t& f)
    {
      point_field_t t = f.c0;
      f.c0 = f.c0 - f.c1;
      f.c1 = f.c1 + t;
    }

    static void mul_fp6_by_nonresidue(fq6_field_t& f)
    {
      g2_point_field_t t = f.c1;
      f.c1 = f.c0;
      f.c0 = f.c2;
      mul_fp2_by_nonresidue(f.c0);
      f.c2 = t;
    }

    static void frobenius_map(fq12_field_t& f, unsigned power)
    {
      fq6_config::frobenius_map(f.c0.c0, f.c0.c1, f.c0.c2, power);
      fq6_config::frobenius_map(f.c1.c0, f.c1.c1, f.c1.c2, power);
      f.c1 *= fq12_config::FROBENIUS_COEFF_C1[power % 12];
    }

    using Fp = point_field_t;          // Base prime field
    using Fp2 = g2_point_field_t;      // Quadratic extension
    using Fp6 = fq6_field_t;           // Cubic extension over Fp2
    using Fp12 = fq12_field_t;         // Quadratic extension over Fp6
    using G1Affine = affine_t;         // G1 group (affine coordinates)
    using G1Projective = projective_t; // G1 group (projective coordinates)
    using G2Affine = g2_affine_t;      // G2 group (affine coordinates)
    using TargetField = Fp12;          // Result of the pairing
    using G2Config = G2;
  };

  // Alias the pairing methods
  using icicle_bls12_pairing::final_exponentiation;
  using icicle_bls12_pairing::miller_loop;
}; // namespace pairing_bls12_381