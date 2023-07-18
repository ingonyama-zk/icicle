#pragma once
#include "../../utils/storage.cuh"

namespace PARAMS_${curve_name_U} {
  struct fp_config {
    static constexpr unsigned limbs_count = ${fp_num_limbs};
    static constexpr unsigned omegas_count = ${num_omegas};
    static constexpr unsigned modulus_bit_count = ${fp_modulus_bit_count};
    
    static constexpr storage<limbs_count> modulus = {${fp_modulus}};
    static constexpr storage<limbs_count> modulus_2 = {${fp_modulus_2}};
    static constexpr storage<limbs_count> modulus_4 = {${fp_modulus_4}};
    static constexpr storage<2*limbs_count> modulus_wide = {${fp_modulus_wide}};
    static constexpr storage<2*limbs_count> modulus_squared = {${fp_modulus_squared}};
    static constexpr storage<2*limbs_count> modulus_squared_2 = {${fp_modulus_squared_2}};
    static constexpr storage<2*limbs_count> modulus_squared_4 = {${fp_modulus_squared_4}};

    static constexpr storage<limbs_count> m = {${fp_m}};
    static constexpr storage<limbs_count> one = {${fp_one}};
    static constexpr storage<limbs_count> zero = {${fp_zero}};

    static constexpr storage_array<omegas_count, limbs_count> omega = { {
        ${omega}
    } };


    static constexpr storage_array<omegas_count, limbs_count> omega_inv = { {
        ${omega_inv}
    } };
    

    static constexpr storage_array<omegas_count, limbs_count> inv = { {
        ${inv}
    } }; 
  };

  struct fq_config {
    static constexpr unsigned limbs_count = ${fq_num_limbs};
    static constexpr unsigned modulus_bit_count = ${fq_modulus_bit_count};
    static constexpr storage<limbs_count> modulus = {${fq_modulus}};
    static constexpr storage<limbs_count> modulus_2 = {${fq_modulus_2}};
    static constexpr storage<limbs_count> modulus_4 = {${fq_modulus_4}};
    static constexpr storage<2*limbs_count> modulus_wide = {${fq_modulus_wide}};
    static constexpr storage<2*limbs_count> modulus_squared = {${fq_modulus_squared}};
    static constexpr storage<2*limbs_count> modulus_squared_2 = {${fq_modulus_squared_2}};
    static constexpr storage<2*limbs_count> modulus_squared_4 = {${fq_modulus_squared_4}};
    static constexpr storage<limbs_count> m = {${fq_m}};
    static constexpr storage<limbs_count> one = {${fq_one}};
    static constexpr storage<limbs_count> zero = {${fq_zero}};
    // i^2, the square of the imaginary unit for the extension field
    static constexpr uint32_t i_squared = 1;
    // true if i^2 is negative
    static constexpr bool i_squared_is_negative = true;
    // G1 and G2 generators 
    static constexpr storage<limbs_count> g1_gen_x = {${fq_gen_x}};
    static constexpr storage<limbs_count> g1_gen_y = {${fq_gen_y}};
    static constexpr storage<limbs_count> g2_gen_x_re = {${fq_gen_x_re}};
    static constexpr storage<limbs_count> g2_gen_x_im = {${fq_gen_x_im}};
    static constexpr storage<limbs_count> g2_gen_y_re = {${fq_gen_y_re}};
    static constexpr storage<limbs_count> g2_gen_y_im = {${fq_gen_y_im}};
  };

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {${weier_b}};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_re = {${weier_b_g2_re}};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_im = {${weier_b_g2_im}};
}
