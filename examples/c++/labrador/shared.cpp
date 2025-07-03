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

std::vector<Rq> compute_Q_poly(size_t n, size_t r, size_t JL_out, const std::byte* seed, size_t seed_len, size_t JL_i)
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

std::vector<Rq> sample_low_norm_challenges(size_t n, size_t r, const std::byte* seed, size_t seed_len)
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
  //     ICICLE_CHECK(check_norm_bound(challenge[i].values, d, eNormType::Lop, OP_NORM_BOUND, {}, &norm_bound));

  //     if (norm_bound) {
  //       break;
  //     } else {
  //       j_ch[i]++;
  //     }
  //   }
  // }
  return challenge;
}

Oracle create_oracle_seed(const std::byte* seed, size_t seed_len, const LabradorInstance& inst)
{
  std::vector<std::byte> buf;

  auto append = [&](auto value) {
    std::byte* p = reinterpret_cast<std::byte*>(&value);
    buf.insert(buf.end(), p, p + sizeof(value));
  };

  // 0. external seed
  buf.insert(buf.end(), seed, seed + seed_len);

  // 1. fixed protocol parameters
  const LabradorParam& prm = inst.param;
  append(prm.r);
  append(prm.n);
  append(prm.kappa);
  append(prm.kappa1);
  append(prm.kappa2);
  append(prm.base1);
  append(prm.base2);
  append(prm.base3);
  append(prm.JL_out);
  append(prm.beta);

  // 1.a Ajtai seed (variable length)
  append(prm.ajtai_seed.size());
  buf.insert(buf.end(), prm.ajtai_seed.begin(), prm.ajtai_seed.end());

  // 2. only counts of constraints
  append(inst.equality_constraints.size());
  append(inst.const_zero_constraints.size());

  // TODO: add contents of equality and const_zero constraints

  return Oracle(buf.data(), buf.size());
}

uint32_t calc_base0(size_t r, uint64_t op_norm_bound, double beta)
{
  uint32_t base0 = sqrt(op_norm_bound * beta * sqrt(r));
  return base0;
}

std::pair<size_t, size_t> compute_mu_nu(size_t n, size_t m)
{
  // setting r_prime^2 = C * n_prime
  float C = 1.0 / 4.0;
  float m_plus_2n = 2 * n + m;
  float frac = pow(C / m_plus_2n / m_plus_2n, 1.0 / 3.0);
  if (n > m) {
    float nu_f = frac * n + 1.0;
    float mu_f = ceil(m * nu_f / n);
    return std::make_pair(size_t(mu_f), size_t(nu_f));
  } else {
    float mu_f = frac * m + 1.0;
    float nu_f = ceil(n * mu_f / m);
    return std::make_pair(size_t(mu_f), size_t(nu_f));
  }
  // size_t nu = 1 << 3, mu = 1 << 3;
}

size_t secure_msis_rank()
{
  const double log_delta = log(1.0045);
  const double log_q = log(get_q<Zq>());

  double k_f = pow(log_q - 1.0, 2) / 4 / log_delta / log_q / Rq::d;
  return ceil(k_f);
}

size_t RecursionPreparer::z0_begin_idx() const { return 0; }

size_t RecursionPreparer::z1_begin_idx() const { return nu * n_prime; }

size_t RecursionPreparer::t_begin_idx() const { return (2 * nu) * n_prime; }

size_t RecursionPreparer::g_begin_idx() const { return (2 * nu + L_t) * n_prime; }

size_t RecursionPreparer::h_begin_idx() const { return (2 * nu + L_t + L_g) * n_prime; }

eIcicleError RecursionPreparer::copy_like_z0(Rq* dst, const Rq* src) const
{
  // copy to dst[z0_begin_idx() : z0_begin_idx() + n]
  return icicle_copy(&dst[z0_begin_idx()], src, prev_n * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_z1(Rq* dst, const Rq* src) const
{
  // copy to dst[z1_begin_idx() : z1_begin_idx() + n]
  return icicle_copy(&dst[z1_begin_idx()], src, prev_n * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_t(Rq* dst, const Rq* src) const
{
  // copy to dst[t_begin_idx() : t_begin_idx() + |t|]
  return icicle_copy(&dst[t_begin_idx()], src, t_len * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_g(Rq* dst, const Rq* src) const
{
  // copy to dst[g_begin_idx() : g_begin_idx() + |g|]
  return icicle_copy(&dst[g_begin_idx()], src, g_len * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_h(Rq* dst, const Rq* src) const
{
  // copy to dst[h_begin_idx() : h_begin_idx() + |h|]
  return icicle_copy(&dst[h_begin_idx()], src, h_len * sizeof(Rq));
}

LabradorInstance prepare_recursion_instance(
  const LabradorParam& prev_param,
  const EqualityInstance& final_const,
  const PartialTranscript& trs,
  uint32_t base0,
  size_t mu,
  size_t nu)
{
  const size_t r = final_const.r;
  const size_t n = final_const.n;
  constexpr size_t d = Rq::d;

  assert(prev_param.r == r);
  assert(prev_param.n == n);

  std::vector<Tq> u1 = trs.prover_msg.u1;
  std::vector<Tq> u2 = trs.prover_msg.u2;
  std::vector<Tq> challenges_hat = trs.challenges_hat;

  RecursionPreparer preparer{prev_param, mu, nu, base0};

  size_t n_prime = preparer.n_prime;
  size_t t_len = preparer.t_len;
  size_t g_len = preparer.g_len;
  size_t h_len = preparer.h_len;
  size_t L_t = preparer.L_t;
  size_t L_g = preparer.L_g;
  size_t L_h = preparer.L_h;
  size_t r_prime = preparer.r_prime;
  // Step 7: Let recursion_instance be a new empty LabradorInstance

  std::vector<std::byte> new_ajtai_seed(prev_param.ajtai_seed);
  new_ajtai_seed.push_back(std::byte('1'));

  // TODO: beta needs to be set correctly for protocol to run
  // currently too large
  double beta = r * n * d * prev_param.beta;
  uint32_t base_prime0 = calc_base0(r_prime, OP_NORM_BOUND, beta);
  LabradorParam recursion_param{
    r_prime,
    n_prime,
    new_ajtai_seed,
    secure_msis_rank(), // kappa
    secure_msis_rank(), // kappa1
    secure_msis_rank(), // kappa2,
    base_prime0,        // base1
    base_prime0,        // base2
    base_prime0,        // base3
    beta,               // beta
  };
  LabradorInstance recursion_instance{recursion_param};

  Zq _zero = Zq::zero();
  Zq two = Zq::from(2);
  Zq two_inv = Zq::inverse(two);
  size_t l3 = icicle::balanced_decomposition::compute_nof_digits<Zq>(prev_param.base3);

  // Step 8: add the equality constraint u1=tB + gC to recursion_instance
  // Generate B, C
  // TODO: change this so that B,C need not be computed and stored
  std::vector<Tq> B(prev_param.t_len() * prev_param.kappa1), C(prev_param.g_len() * prev_param.kappa1),
    B_t(prev_param.kappa1 * prev_param.t_len()), C_t(prev_param.kappa1 * prev_param.g_len());

  std::vector<std::byte> seed_B(prev_param.ajtai_seed), seed_C(prev_param.ajtai_seed);
  seed_B.push_back(std::byte('1'));
  seed_C.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(seed_B.data(), seed_B.size(), false, {}, B.data(), B.size()));
  ICICLE_CHECK(random_sampling(seed_C.data(), seed_C.size(), false, {}, C.data(), C.size()));

  // B_t, C_t are transposed B, C
  ICICLE_CHECK(matrix_transpose<Tq>(B.data(), prev_param.t_len(), prev_param.kappa1, {}, B_t.data()));
  ICICLE_CHECK(matrix_transpose<Tq>(C.data(), prev_param.g_len(), prev_param.kappa1, {}, C_t.data()));

  // negate u1
  ICICLE_CHECK(
    scalar_sub_vec(&_zero, reinterpret_cast<Zq*>(u1.data()), d * u1.size(), {}, reinterpret_cast<Zq*>(u1.data())));

  for (size_t i = 0; i < prev_param.kappa1; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    ICICLE_CHECK(preparer.copy_like_t(new_constraint.phi.data(), &B_t[i * t_len]));
    ICICLE_CHECK(preparer.copy_like_g(new_constraint.phi.data(), &C_t[i * g_len]));

    new_constraint.b = u1[i];

    // TESTING: at this point you can check whether the witness satisfies the constraint

    recursion_instance.add_equality_constraint(new_constraint);
  }

  // The vectors B, C, B_t, C_t are no longer needed, so we delete them to free memory.
  B.clear();
  B.shrink_to_fit();
  C.clear();
  C.shrink_to_fit();
  B_t.clear();
  B_t.shrink_to_fit();
  C_t.clear();
  C_t.shrink_to_fit();

  // Step 9: add the equality constraint u2=hD to recursion_instance
  // Generate D
  // TODO: change this so that D need not be computed and stored
  std::vector<Tq> D(h_len * prev_param.kappa2), D_t(prev_param.kappa2 * h_len);

  std::vector<std::byte> seed_D(prev_param.ajtai_seed);
  seed_D.push_back(std::byte('3'));
  ICICLE_CHECK(random_sampling(seed_D.data(), seed_D.size(), false, {}, D.data(), D.size()));
  ICICLE_CHECK(matrix_transpose<Tq>(D.data(), h_len, prev_param.kappa2, {}, D_t.data()));

  // negate u2
  ICICLE_CHECK(
    scalar_sub_vec(&_zero, reinterpret_cast<Zq*>(u2.data()), d * u2.size(), {}, reinterpret_cast<Zq*>(u2.data())));

  for (size_t i = 0; i < prev_param.kappa2; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    ICICLE_CHECK(preparer.copy_like_h(new_constraint.phi.data(), &D_t[i * h_len]));

    new_constraint.b = u2[i];

    // TESTING: at this point you can check whether the witness satisfies the constraint

    recursion_instance.add_equality_constraint(new_constraint);
  }
  // The vectors D, D_t are no longer needed, so we delete them to free memory.
  D.clear();
  D.shrink_to_fit();
  D_t.clear();
  D_t.shrink_to_fit();

  // Step 10: add the equality constraint Az - sum_i c_i t_i =0 to recursion_instance
  // Generate A
  // TODO: change this so that A need not be computed and stored
  size_t kappa = prev_param.kappa;
  size_t l1 = icicle::balanced_decomposition::compute_nof_digits<Zq>(prev_param.base1);
  std::vector<Tq> A(n * kappa);

  std::vector<std::byte> seed_A(prev_param.ajtai_seed);
  seed_A.push_back(std::byte('0'));
  ICICLE_CHECK(random_sampling(seed_A.data(), seed_A.size(), false, {}, A.data(), n * kappa));

  // A transpose
  std::vector<Tq> A_t(kappa * n);
  ICICLE_CHECK(matrix_transpose<Tq>(A.data(), n, kappa, {}, A_t.data()));

  for (size_t i = 0; i < kappa; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    ICICLE_CHECK(preparer.copy_like_z0(new_constraint.phi.data(), &A_t[i * n]));
    ICICLE_CHECK(preparer.copy_like_z1(new_constraint.phi.data(), &A_t[i * n]));
    // new_constraint.phi[nu+ j] = base0*new_constraint.phi[nu+ j]
    Zq base0_scalar = Zq::from(base0);
    ICICLE_CHECK(scalar_mul_vec(
      &base0_scalar, reinterpret_cast<const Zq*>(&new_constraint.phi[preparer.z1_begin_idx()]), n * d, {},
      reinterpret_cast<Zq*>(&new_constraint.phi[preparer.z1_begin_idx()])));

    // TODO: think about vectorising
    std::vector<Tq> temp(r * kappa, zero());
    for (size_t j = 0; j < r; j++) {
      Tq neg_challenge_hat_j = challenges_hat[j];
      scalar_sub_vec(&_zero, challenges_hat[j].values, d, {}, neg_challenge_hat_j.values);
      temp[j * kappa + i] = neg_challenge_hat_j;
    }
    // construct the vector t_mul = [temp | base1*temp | base1^2* temp | ... | base1^l1 * temp]
    std::vector<Tq> t_mul(t_len); // t_len == l1 * r * kappa

    Zq b1_zq = Zq::from(prev_param.base1);
    ICICLE_CHECK(icicle_copy(&t_mul[0], temp.data(), temp.size() * sizeof(Tq)));
    for (size_t j = 1; j < l1; j++) {
      // temp = base1*temp
      scalar_mul_vec(
        &b1_zq, reinterpret_cast<Zq*>(temp.data()), temp.size() * d, {}, reinterpret_cast<Zq*>(temp.data()));
      // t_mul[j * r * kappa: ] = temp
      ICICLE_CHECK(icicle_copy(&t_mul[j * r * kappa], temp.data(), temp.size() * sizeof(Tq)));
    }

    ICICLE_CHECK(preparer.copy_like_t(new_constraint.phi.data(), t_mul.data()));

    recursion_instance.add_equality_constraint(new_constraint);
  }

  std::vector<Tq> c_times_ct(r * r);
  /* Step 11: add c^t * Phi * z - c^t H c == 0 */ {
    EqualityInstance step11_constraint(r_prime, n_prime);
    std::vector<Tq> c_times_phi(n);
    // c_times_phi = c^t * Phi
    ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, final_const.phi.data(), r, n, {}, c_times_phi.data()));
    ICICLE_CHECK(preparer.copy_like_z0(step11_constraint.phi.data(), c_times_phi.data()));
    Zq b0_zq = Zq::from(base0);
    // c_times_phi = base0 * c_times_phi
    scalar_mul_vec(
      &b0_zq, reinterpret_cast<Zq*>(c_times_phi.data()), c_times_phi.size() * d, {},
      reinterpret_cast<Zq*>(c_times_phi.data()));

    ICICLE_CHECK(preparer.copy_like_z1(step11_constraint.phi.data(), c_times_phi.data()));

    ICICLE_CHECK(matmul(challenges_hat.data(), r, 1, challenges_hat.data(), 1, r, {}, c_times_ct.data()));
    // c_times_ct = 2 * c_times_ct
    scalar_mul_vec(
      &two, reinterpret_cast<Zq*>(c_times_ct.data()), c_times_ct.size() * d, {},
      reinterpret_cast<Zq*>(c_times_ct.data()));
    // rescale diagonal back to original
    for (size_t j = 0; j < r; j++) {
      scalar_mul_vec(&two_inv, c_times_ct[j * r + j].values, d, {}, c_times_ct[j * r + j].values);
    }
    // now c_times_ct[j,j] =challenges_hat[j]^2 and
    // for j!=k c_times_ct[j,k] = 2* challenges_hat[j] * challenges_hat[k]
    std::vector<Tq> temp = extract_symm_part(c_times_ct.data(), r);

    // negate temp
    ICICLE_CHECK(scalar_sub_vec(
      &_zero, reinterpret_cast<Zq*>(temp.data()), d * temp.size(), {}, reinterpret_cast<Zq*>(temp.data())));

    // construct the vector h_mul = [temp | base3*temp | base3^2* temp | ... | base3^l3 * temp]
    std::vector<Tq> h_mul(h_len);
    size_t l3 = icicle::balanced_decomposition::compute_nof_digits<Zq>(prev_param.base3);

    Zq b3_zq = Zq::from(prev_param.base3);
    ICICLE_CHECK(icicle_copy(&h_mul[0], temp.data(), temp.size() * sizeof(Tq)));
    for (size_t j = 1; j < l3; j++) {
      // temp = base3*temp
      scalar_mul_vec(
        &b3_zq, reinterpret_cast<Zq*>(temp.data()), temp.size() * d, {}, reinterpret_cast<Zq*>(temp.data()));
      // h_mul[j * temp.size(): ] = temp
      ICICLE_CHECK(icicle_copy(&h_mul[j * temp.size()], temp.data(), temp.size() * sizeof(Tq)));
    }

    ICICLE_CHECK(preparer.copy_like_h(step11_constraint.phi.data(), h_mul.data()));

    recursion_instance.add_equality_constraint(step11_constraint);
  }

  // returns a constant polynomial in Tq
  auto const_poly = [](const Zq& c) {
    Tq poly;
    for (size_t j = 0; j < d; j++) {
      poly.values[j] = c;
    }
    return poly;
  };

  Tq poly_one = const_poly(Zq::one());
  size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(prev_param.base2);

  /* Step 12: \sum_ij a_ij G_ij + \sum_i h_ii + b == 0 */ {
    EqualityInstance step12_constraint(r_prime, n_prime);

    // construct matrix M such that M[i,j] = final_const.a[i,j] + final_const.a[j,i] for i != j else M[i,i] =
    // final_const.a[i,i]
    std::vector<Tq> M(r * r);
    // M = final_const.a^t
    ICICLE_CHECK(matrix_transpose(final_const.a.data(), r, r, {}, M.data()));
    // M = final_const.a + final_const.a^t
    ICICLE_CHECK(vector_add(final_const.a.data(), M.data(), r * r, {}, M.data()));
    // rescale diagonal back to original
    for (size_t j = 0; j < r; j++) {
      scalar_mul_vec(&two_inv, M[j * r + j].values, d, {}, M[j * r + j].values);
    }
    // extract the symmetric part of M as vector a_symm
    std::vector<Tq> a_symm = extract_symm_part(M.data(), r);

    // construct the vector g_mul = [a_symm | base2*a_symm | base2^2* a_symm | ... | base2^l2 * a_symm]
    std::vector<Tq> g_mul(g_len);
    Zq b2_zq = Zq::from(prev_param.base2);
    ICICLE_CHECK(icicle_copy(&g_mul[0], a_symm.data(), a_symm.size() * sizeof(Tq)));
    for (size_t j = 1; j < l2; j++) {
      // a_symm = base2 * a_symm
      scalar_mul_vec(
        &b2_zq, reinterpret_cast<Zq*>(a_symm.data()), a_symm.size() * d, {}, reinterpret_cast<Zq*>(a_symm.data()));
      // g_mul[j * r * kappa: ] = a_symm
      ICICLE_CHECK(icicle_copy(&g_mul[j * a_symm.size()], a_symm.data(), a_symm.size() * sizeof(Tq)));
    }

    ICICLE_CHECK(preparer.copy_like_g(step12_constraint.phi.data(), g_mul.data()));

    // construct the vector symm_I = symmetric_part(I)
    size_t r_choose_2 = (r * (r + 1)) / 2;
    std::vector<Tq> symm_I(r_choose_2, zero());
    size_t i = 0;
    size_t skip = r;

    while (i < r_choose_2) {
      ICICLE_CHECK(icicle_copy(symm_I[i].values, poly_one.values, d * sizeof(Zq)));
      i += skip;
      skip--;
    }

    std::vector<Tq> h_mul(h_len);
    Zq b3_zq = Zq::from(prev_param.base3);
    // [symm_I | base3*symm_I | base3^2* symm_I | ... | base3^l3 * symm_I]
    ICICLE_CHECK(icicle_copy(&h_mul[0], symm_I.data(), symm_I.size() * sizeof(Tq)));
    for (size_t j = 1; j < l3; j++) {
      // symm_I = base3*symm_I
      scalar_mul_vec(
        &b3_zq, reinterpret_cast<Zq*>(symm_I.data()), symm_I.size() * d, {}, reinterpret_cast<Zq*>(symm_I.data()));
      // h_mul[j * symm_I.size(): ] = symm_I
      ICICLE_CHECK(icicle_copy(&h_mul[j * symm_I.size()], symm_I.data(), symm_I.size() * sizeof(Tq)));
    }
    ICICLE_CHECK(preparer.copy_like_h(step12_constraint.phi.data(), h_mul.data()));

    step12_constraint.b = final_const.b;
    recursion_instance.add_equality_constraint(step12_constraint);
  }

  /* Step 13: <z, z> - sum_ij c_i c_j G_ij == 0 */ {
    EqualityInstance step13_constraint(r_prime, n_prime);

    Tq b0_poly = const_poly(Zq::from(base0));
    Tq b0_sq_poly = const_poly(Zq::from(base0 * base0));
    for (size_t i = 0; i < nu; i++) {
      // a[i,i] = 1
      step13_constraint.a[i * r_prime + i] = poly_one;
      // a[i+nu, i+nu] = base0^2
      step13_constraint.a[(i + nu) * r_prime + (i + nu)] = b0_sq_poly;
      // a[i, i+nu] = base0
      step13_constraint.a[(i + nu) * r_prime + i] = b0_poly;
      // a[i+nu, i] = base0
      step13_constraint.a[i * r_prime + i + nu] = b0_poly;
    }

    std::vector<Tq> temp = extract_symm_part(c_times_ct.data(), r);

    ICICLE_CHECK(scalar_sub_vec(
      &_zero, reinterpret_cast<Zq*>(temp.data()), d * temp.size(), {}, reinterpret_cast<Zq*>(temp.data())));

    // construct the vector g_mul2 = [temp | base2*temp | base2^2* temp | ... | base3^l2 * temp]
    std::vector<Tq> g_mul(g_len);
    Zq b2_zq = Zq::from(prev_param.base2);
    ICICLE_CHECK(icicle_copy(&g_mul[0], temp.data(), temp.size() * sizeof(Tq)));
    for (size_t j = 1; j < l2; j++) {
      // temp = base2 * temp
      scalar_mul_vec(
        &b2_zq, reinterpret_cast<Zq*>(temp.data()), temp.size() * d, {}, reinterpret_cast<Zq*>(temp.data()));
      // g_mul2[j * temp.size(): ] = temp
      ICICLE_CHECK(icicle_copy(&g_mul[j * temp.size()], temp.data(), temp.size() * sizeof(Tq)));
    }

    ICICLE_CHECK(preparer.copy_like_g(step13_constraint.phi.data(), g_mul.data()));

    recursion_instance.add_equality_constraint(step13_constraint);
  }
  // Step 14: already done

  return recursion_instance;
}