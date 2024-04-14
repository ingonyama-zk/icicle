#pragma once

#include "fields/storage.cuh"
#include "fields/field.cuh"

#ifdef EXT_FIELD
#include "fields/quartic_extension.cuh"
#endif

namespace baby_bear {
  struct fp_config {
    static constexpr unsigned limbs_count = 1;
    static constexpr unsigned omegas_count = 28;
    static constexpr unsigned modulus_bit_count = 31;
    static constexpr unsigned num_of_reductions = 1;

    static constexpr storage<limbs_count> modulus = {0x78000001};
    static constexpr storage<limbs_count> modulus_2 = {0xf0000002};
    static constexpr storage<limbs_count> modulus_4 = {0x00000000};
    static constexpr storage<limbs_count> neg_modulus = {0x87ffffff};
    static constexpr storage<2 * limbs_count> modulus_wide = {0x78000001, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {0xf0000001, 0x38400000};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {0xe0000002, 0x70800001};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {0xc0000004, 0xe1000003};

    static constexpr storage<limbs_count> m = {0x88888887};
    static constexpr storage<limbs_count> one = {0x00000001};
    static constexpr storage<limbs_count> zero = {0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xffffffe};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x38400000};

    static constexpr storage_array<omegas_count, limbs_count> omega = {
      {{0x78000000}, {0x10faa3e0}, {0x6b615c47}, {0x21ceed5a}, {0x2c1c3348}, {0x36c54c86}, {0x701dd01c},
       {0x56a9a28e}, {0x03e4cabf}, {0x5bacde79}, {0x1eb53838}, {0x1cd781af}, {0x0961a0b7}, {0x65098a87},
       {0x77851a0b}, {0x5bcba331}, {0x053fc0f5}, {0x5bf816e5}, {0x4bb124ab}, {0x571e9d4e}, {0x313732cb},
       {0x28aca172}, {0x4e319b52}, {0x45692d95}, {0x14ff4ba1}, {0x00004951}, {0x00000089}}};

    static constexpr storage_array<omegas_count, limbs_count> omega_inv = {
      {{0x78000000}, {0x67055c21}, {0x5ee99486}, {0x0bb4c4e4}, {0x4ab33b27}, {0x044b4497}, {0x410e23aa},
       {0x08a7ee2b}, {0x563cb93d}, {0x3d70b4b7}, {0x77d999f1}, {0x6ceb65b5}, {0x49e7f635}, {0x0eae3a8c},
       {0x238b8a78}, {0x70d71b0a}, {0x0eaacc45}, {0x5af0f193}, {0x47303308}, {0x573cbfad}, {0x29ff72c0},
       {0x05af9dac}, {0x00ef24df}, {0x26985530}, {0x22d1ce4b}, {0x08359375}, {0x2cabe994}}};

    static constexpr storage_array<omegas_count, limbs_count> inv = {
      {{0x3c000001}, {0x5a000001}, {0x69000001}, {0x70800001}, {0x74400001}, {0x76200001}, {0x77100001},
       {0x77880001}, {0x77c40001}, {0x77e20001}, {0x77f10001}, {0x77f88001}, {0x77fc4001}, {0x77fe2001},
       {0x77ff1001}, {0x77ff8801}, {0x77ffc401}, {0x77ffe201}, {0x77fff101}, {0x77fff881}, {0x77fffc41},
       {0x77fffe21}, {0x77ffff11}, {0x77ffff89}, {0x77ffffc5}, {0x77ffffe3}, {0x77fffff2}}};

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 11;
    // true if nonresidue is negative.
    // TODO: we're very confused by plonky3 and risc0 having different nonresidues: 11 and -11 respectively
    static constexpr bool nonresidue_is_negative = true;
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;

#ifdef EXT_FIELD
  /**
   * Extension field of `scalar_t` enabled if `-DEXT_FIELD` env variable is.
   */
  typedef ExtensionField<fp_config> extension_t;
#endif
} // namespace baby_bear
