#include "shared.h"

std::vector<Tq> ajtai_commitment(
  const std::byte* ajtai_mat_seed, size_t seed_len, size_t input_len, size_t output_len, const Tq* S, size_t S_len)
{
  size_t batch_size = S_len / input_len;
  // Assert that data_len is a multiple of input_len
  assert(batch_size * input_len == S_len);
  // TODO: change this so that A need not be computed and stored
  std::vector<Tq> A(input_len * output_len);
  ICICLE_CHECK(random_sampling(ajtai_mat_seed, seed_len, false, {}, A.data(), A.size()));

  std::vector<Tq> comm(batch_size * output_len);
  ICICLE_CHECK(matmul(S, batch_size, input_len, A.data(), input_len, output_len, {}, comm.data()));
  return comm;
}

std::vector<Rq> compute_Q_poly(size_t n, size_t r, size_t JL_out, std::byte* seed, size_t seed_len, size_t JL_i)
{
  size_t d = Rq::d;

  // Step 17: Create conjugated polynomial vectors from JL matrix rows
  std::vector<Rq> Q(JL_out * r * n);
  // Create seed for P matrix (same as in step 12)
  std::vector<std::byte> jl_seed(seed, seed + seed_len);
  jl_seed.push_back(std::byte(JL_i));

  // compute the Pi matrix, conjugated in Rq
  ICICLE_CHECK(get_jl_matrix_rows<Rq>(
    jl_seed.data(), jl_seed.size(),
    r * n,  // row_size
    0,      // row_index
    JL_out, // num_rows
    true,   // conjugate
    {},     // config
    Q.data()));

  return Q;
}

// TODO: Simply returns the polynomial x for every challenge rn
std::vector<Rq> sample_low_norm_challenges(size_t n, size_t r, std::byte* seed, size_t seed_len)
{
  size_t d = Rq::d;
  std::vector<Rq> challenge(r, zero());
  for (auto& c : challenge) {
    c.values[1] = Zq::from(1);
  }
  // std::vector<size_t> j_ch(r, 0);
  // // TODO: can parallelise the i loop
  // for (size_t i = 0; i < r; i++) {
  //   while (true) {
  //     std::vector<std::byte> ch_seed(seed, seed + seed_len);
  //     ch_seed.push_back(std::byte(i));
  //     ch_seed.push_back(std::byte(j_ch[i]));
  //     ICICLE_CHECK(sample_challenge_polynomials(ch_seed.data(), ch_seed.size(), {1, 2}, {31, 10}, challenge[i]));

  //     bool norm_bound = false;
  //     ICICLE_CHECK(check_norm_bound(challenge[i].values, d, eNormType::Lop, 15, {}, &norm_bound));

  //     if (norm_bound) {
  //       break;
  //     } else {
  //       j_ch[i]++;
  //     }
  //   }
  // }
  return challenge;
}

LabradorInstance prepare_recursion_instance(
  const LabradorParam& prev_param,
  const EqualityInstance& final_const,
  const PartialTranscript& trs,
  size_t base0,
  size_t mu,
  size_t nu)
{
  const size_t r = final_const.r;
  const size_t n = final_const.n;
  constexpr size_t d = Rq::d;

  std::vector<Tq> u1 = trs.u1;
  std::vector<Tq> u2 = trs.u2;
  std::vector<Tq> challenges_hat = trs.challenges_hat;

  // TODO: check whether these are called correctly
  size_t t_len = prev_param.t_len();
  size_t g_len = prev_param.g_len();
  size_t h_len = prev_param.h_len();

  size_t m = t_len + g_len + h_len;
  size_t n_prime = std::max(std::ceil((double)n / nu), std::ceil((double)m / mu));
  size_t L_t = (t_len + n_prime - 1) / n_prime;
  size_t L_g = (g_len + n_prime - 1) / n_prime;
  size_t L_h = (h_len + n_prime - 1) / n_prime;

  // Step 7: Let recursion_instance be a new empty LabradorInstance
  size_t r_prime = 2 * nu + L_t + L_g + L_h;
  std::vector<std::byte> new_ajtai_seed(prev_param.ajtai_seed);
  new_ajtai_seed.push_back(std::byte('1'));
  // TODO: figure out param using Lattirust code
  LabradorParam recursion_param{
    r_prime,
    n_prime,
    new_ajtai_seed,
    prev_param.kappa,      // kappa
    prev_param.kappa1,     // kappa1
    prev_param.kappa2,     // kappa2,
    prev_param.base1,      // base1
    prev_param.base2,      // base2
    prev_param.base3,      // base3
    100 * prev_param.beta, // beta
  };
  LabradorInstance recursion_instance{recursion_param};

  // Step 8: add the equality constraint u1=Bt + Cg to recursion_instance
  // Generate B, C
  // TODO: change this so that B,C need not be computed and stored
  size_t l1 = icicle::balanced_decomposition::compute_nof_digits<Zq>(prev_param.base1);
  size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(prev_param.base2);
  size_t r_choose_2 = (r * (r + 1)) / 2;
  std::vector<Tq> B(prev_param.kappa1 * l1 * r * prev_param.kappa), C(prev_param.kappa1 * r_choose_2 * l2),
    B_t(prev_param.kappa1 * l1 * r * prev_param.kappa), C_t(prev_param.kappa1 * r_choose_2 * l2);

  std::vector<std::byte> seed_B(prev_param.ajtai_seed), seed_C(prev_param.ajtai_seed);
  seed_B.push_back(std::byte('1'));
  seed_C.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(seed_B.data(), seed_B.size(), false, {}, B.data(), B.size()));
  ICICLE_CHECK(random_sampling(seed_C.data(), seed_C.size(), false, {}, C.data(), C.size()));

  // B_t, C_t are transposed B, C
  ICICLE_CHECK(matrix_transpose<Tq>(B.data(), l1 * r * prev_param.kappa, prev_param.kappa1, {}, B_t.data()));
  ICICLE_CHECK(matrix_transpose<Tq>(C.data(), r_choose_2 * l2, prev_param.kappa1, {}, C_t.data()));

  for (size_t i = 0; i < prev_param.kappa1; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    icicle_copy(&new_constraint.phi[2 * nu * n_prime], &B_t[i * t_len], t_len * sizeof(Tq));
    icicle_copy(&new_constraint.phi[(2 * nu + L_t) * n_prime], &C_t[i * g_len], g_len * sizeof(Tq));

    new_constraint.b = u1[i];

    recursion_instance.add_equality_constraint(new_constraint);
  }

  // Step 9: add the equality constraint u2=Dh to recursion_instance
  // Generate D
  // TODO: change this so that D need not be computed and stored
  size_t l3 = icicle::balanced_decomposition::compute_nof_digits<Zq>(prev_param.base3);
  std::vector<Tq> D(prev_param.kappa2 * l3 * r_choose_2), D_t(prev_param.kappa2 * l3 * r_choose_2);

  std::vector<std::byte> seed_D(prev_param.ajtai_seed);
  seed_D.push_back(std::byte('3'));
  ICICLE_CHECK(random_sampling(seed_D.data(), seed_D.size(), false, {}, D.data(), D.size()));
  ICICLE_CHECK(matrix_transpose<Tq>(D.data(), r_choose_2 * l3, prev_param.kappa2, {}, D_t.data()));

  for (size_t i = 0; i < prev_param.kappa2; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    icicle_copy(&new_constraint.phi[(2 * nu + L_t + L_g) * n_prime], &D_t[i * h_len], sizeof(Tq) * h_len);
    new_constraint.b = u2[i];

    recursion_instance.add_equality_constraint(new_constraint);
  }

  // Step 10: add the equality constraint Az - sum_i c_i t_i =0 to recursion_instance
  // Generate A
  // TODO: change this so that A need not be computed and stored
  std::vector<Tq> A(n * prev_param.kappa);

  std::vector<std::byte> seed_A(prev_param.ajtai_seed);
  seed_A.push_back(std::byte('0'));
  ICICLE_CHECK(random_sampling(seed_A.data(), seed_A.size(), false, {}, A.data(), n * prev_param.kappa));

  // A transpose
  std::vector<Tq> A_t(prev_param.kappa * n);
  ICICLE_CHECK(matrix_transpose<Tq>(A.data(), n, prev_param.kappa, {}, A_t.data()));

  for (size_t i = 0; i < prev_param.kappa; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    icicle_copy(new_constraint.phi.data(), &A_t[i * n], n * sizeof(Tq));
    icicle_copy(&new_constraint.phi[nu * n_prime], &A_t[i * n], n * sizeof(Tq));
    // new_constraint.phi[nu+ j] = base0*new_constraint.phi[nu+ j]
    Zq base0_scalar = Zq::from(base0);
    ICICLE_CHECK(scalar_mul_vec(
      &base0_scalar, reinterpret_cast<const Zq*>(&new_constraint.phi[nu]), n * d, {},
      reinterpret_cast<Zq*>(&new_constraint.phi[nu])));

    // Step 10.d
    size_t k1 = 2 * nu, k2 = 0;
    for (size_t i1 = 0; i1 < r; i1++) {
      for (size_t i2 = 0; i2 < n; i2++) {
        for (size_t i3 = 0; i3 < l1; i3++) {
          if (i2 == i) {
            Tq temp;
            Zq base1_pow = Zq::from(static_cast<int64_t>(std::pow(prev_param.base1, i3)));
            ICICLE_CHECK(scalar_mul_vec(&base1_pow, challenges_hat[i2].values, d, {}, temp.values));
            new_constraint.phi[k1 * n_prime + k2] = temp;
          }
          k2++;
          if (k2 == n_prime) {
            k2 = 0;
            k1++;
          }
        }
      }
    }

    recursion_instance.add_equality_constraint(new_constraint);
  }

  // Step 11:
  EqualityInstance step11_constraint(r_prime, n_prime);
  std::vector<Tq> c_times_phi(n);
  // TODO: vectorize
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < n; j++) {
      Tq temp;
      ICICLE_CHECK(vector_mul(challenges_hat[i].values, final_const.phi[i * n_prime + j].values, d, {}, temp.values));
      ICICLE_CHECK(vector_add(c_times_phi[j].values, temp.values, d, {}, c_times_phi[j].values));
    }
  }
  icicle_copy(step11_constraint.phi.data(), c_times_phi.data(), c_times_phi.size() * sizeof(Tq));
  icicle_copy(&step11_constraint.phi[nu], c_times_phi.data(), c_times_phi.size() * sizeof(Tq));

  // step11_constraint.phi[nu+ j] = base0*step11_constraint.phi[nu+ j]
  Zq base0_scalar = Zq::from(base0);
  ICICLE_CHECK(scalar_mul_vec(
    &base0_scalar, reinterpret_cast<const Zq*>(&step11_constraint.phi[nu]), n * d, {},
    reinterpret_cast<Zq*>(&step11_constraint.phi[nu])));

  size_t s11_k1 = 2 * nu + L_t + L_g, s11_k2 = 0;
  for (size_t i1 = 0; i1 < r; i1++) {
    for (size_t i2 = i1; i2 < r; i2++) {
      for (size_t i3 = 0; i3 < l3; i3++) {
        Tq temp;
        ICICLE_CHECK(vector_mul(challenges_hat[i1].values, challenges_hat[i2].values, d, {}, temp.values));
        Zq minus_1 = Zq::neg(Zq::from(1));
        Zq multiplier = minus_1 * Zq::from(std::pow(prev_param.base3, i3));
        if (i1 != i2) { multiplier = Zq::from(2) * multiplier; }
        ICICLE_CHECK(scalar_mul_vec(&multiplier, temp.values, d, {}, temp.values));
        step11_constraint.phi[s11_k1 * n_prime + s11_k2] = temp;
        s11_k2++;
        if (s11_k2 == n_prime) {
          s11_k2 = 0;
          s11_k1++;
        }
      }
    }
  }
  recursion_instance.add_equality_constraint(step11_constraint);

  // Step 12:
  EqualityInstance step12_constraint(r_prime, n_prime);

  size_t s12_k1 = 2 * nu + L_t, s12_k2 = 0;
  for (size_t i1 = 0; i1 < r; i1++) {
    for (size_t i2 = i1; i2 < r; i2++) {
      for (size_t i3 = 0; i3 < l2; i3++) {
        Tq temp = final_const.a[i1 * n + i2];
        Zq multiplier = Zq::from(std::pow(prev_param.base2, i3));
        if (i1 != i2) { multiplier = Zq::from(2) * multiplier; }

        ICICLE_CHECK(scalar_mul_vec(&multiplier, temp.values, d, {}, temp.values));
        step12_constraint.phi[s12_k1 * n_prime + s12_k2] = temp;
        s12_k2++;
        if (s12_k2 == n_prime) {
          s12_k2 = 0;
          s12_k1++;
        }
      }
    }
  }
  s12_k1 = 2 * nu + L_t + L_g, s12_k2 = 0;
  for (size_t i1 = 0; i1 < r; i1++) {
    for (size_t i2 = i1; i2 < r; i2++) {
      for (size_t i3 = 0; i3 < l3; i3++) {
        if (i1 == i2) {
          Tq temp;
          Zq multiplier = Zq::from(std::pow(prev_param.base3, i3));
          for (size_t k = 0; k < d; k++) {
            temp.values[k] = multiplier;
          }
          step12_constraint.phi[s12_k1 * n_prime + s12_k2] = temp;
        }
        s12_k2++;
        if (s12_k2 == n_prime) {
          s12_k2 = 0;
          s12_k1++;
        }
      }
    }
  }
  step12_constraint.b = final_const.b;
  recursion_instance.add_equality_constraint(step12_constraint);

  // Step 13:
  EqualityInstance step13_constraint(r_prime, n_prime);

  for (int i1 = 0; i1 < 2 * nu; i1++) {
    for (int i2 = 0; i2 < 2 * nu; i2++) {
      Zq c = Zq::from(0);
      if (i1 == i2) {
        if (i1 < nu) {
          c = Zq::from(1);
        } else {
          c = Zq::from(base0 * base0);
        }
      } else if (abs(static_cast<int>(i2 - i1)) == nu) {
        c = Zq::from(2 * base0);
      }
      Tq temp;
      for (size_t k = 0; k < d; k++) {
        temp.values[k] = c;
      }
      step13_constraint.a[i1 * n + i2] = temp;
    }
  }

  size_t s13_k1 = 2 * nu + L_t, s13_k2 = 0;
  for (size_t i1 = 0; i1 < r; i1++) {
    for (size_t i2 = i1; i2 < r; i2++) {
      for (size_t i3 = 0; i3 < l2; i3++) {
        Tq temp;
        ICICLE_CHECK(vector_mul(challenges_hat[i1].values, challenges_hat[i2].values, d, {}, temp.values));
        Zq minus_1 = Zq::neg(Zq::from(1));
        Zq multiplier = minus_1 * Zq::from(std::pow(prev_param.base2, i3));
        if (i1 != i2) { multiplier = Zq::from(2) * multiplier; }
        ICICLE_CHECK(scalar_mul_vec(&multiplier, temp.values, d, {}, temp.values));
        step11_constraint.phi[s13_k1 * n_prime + s13_k2] = temp;
        s13_k2++;
        if (s13_k2 == n_prime) {
          s13_k2 = 0;
          s13_k1++;
        }
      }
    }
  }
  recursion_instance.add_equality_constraint(step11_constraint);

  // Step 14: already done

  return recursion_instance;
}
