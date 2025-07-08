#include "verifier.h"

// Fills up trs correctly assuming trs.prover_msg are correctly filled
void LabradorBaseVerifier::create_transcript()
{
  const size_t d = Rq::d;
  // 1. seed1 from u1
  const auto& u1 = trs.prover_msg.u1;
  const std::byte* u1_bytes = reinterpret_cast<const std::byte*>(u1.data());
  size_t u1_len = u1.size() * sizeof(Tq);
  trs.seed1 = oracle.generate(u1_bytes, u1_len);

  // 2. seed2 from JL_i and p
  size_t JL_i = trs.prover_msg.JL_i;
  const std::vector<Zq>& p = trs.prover_msg.p;
  std::vector<std::byte> jl_buf(sizeof(size_t));
  std::memcpy(jl_buf.data(), &JL_i, sizeof(size_t));
  jl_buf.insert(
    jl_buf.end(), reinterpret_cast<const std::byte*>(p.data()),
    reinterpret_cast<const std::byte*>(p.data()) + p.size() * sizeof(Zq));
  trs.seed2 = oracle.generate(jl_buf.data(), jl_buf.size());

  // 3. psi and omega sampling
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t JL_out = lab_inst.param.JL_out;
  const size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;
  trs.psi.resize(num_aggregation_rounds * L);
  trs.omega.resize(num_aggregation_rounds * JL_out);
  // psi seed = seed2 || 0x01
  std::vector<std::byte> psi_seed(trs.seed2);
  psi_seed.push_back(std::byte('1'));
  if (trs.psi.size() > 0) {
    ICICLE_CHECK(random_sampling(trs.psi.size(), true, psi_seed.data(), psi_seed.size(), {}, trs.psi.data()));
  }
  // omega seed = seed2 || 0x02
  std::vector<std::byte> omega_seed(trs.seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(trs.omega.size(), true, omega_seed.data(), omega_seed.size(), {}, trs.omega.data()));

  // 4. seed3 from msg3 (b_agg)
  const auto& msg3 = trs.prover_msg.b_agg;
  const std::byte* msg3_bytes = reinterpret_cast<const std::byte*>(msg3.data());
  size_t msg3_len = msg3.size() * sizeof(Tq);
  trs.seed3 = oracle.generate(msg3_bytes, msg3_len);

  // 5. alpha_hat sampling
  // After we aggregate the L const-zero constraints the instance will
  // have `num_aggregation_rounds` extra EqualityInstances, so we must sample
  // α for the *final* size in advance.
  const size_t K = lab_inst.equality_constraints.size() + num_aggregation_rounds;
  assert(K > 0);

  trs.alpha_hat.resize(K);
  std::vector<std::byte> alpha_seed(trs.seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(K, true, alpha_seed.data(), alpha_seed.size(), {}, trs.alpha_hat.data()));

  // 6. seed4 from u2
  const auto& u2 = trs.prover_msg.u2;
  const std::byte* u2_bytes = reinterpret_cast<const std::byte*>(u2.data());
  size_t u2_bytes_len = u2.size() * sizeof(Tq);
  trs.seed4 = oracle.generate(u2_bytes, u2_bytes_len);

  // 7. challenges_hat sampling
  size_t n = lab_inst.param.n;
  size_t r = lab_inst.param.r;
  std::vector<Rq> challenge = sample_low_norm_challenges(n, r, trs.seed4.data(), trs.seed4.size());
  trs.challenges_hat.resize(r);
  ntt(challenge.data(), challenge.size(), NTTDir::kForward, {}, trs.challenges_hat.data());
}

// Chunked LInfinity norm check to support vectors larger than 2^16 elements.
// Uses the same API as `check_norm_bound` except the norm type is fixed to LInfinity.
inline eIcicleError check_norm_bound_LInfinity_chunked(
  const Zq* input, size_t size, uint64_t norm_bound, const VecOpsConfig& cfg, bool* output)
{
  if (output == nullptr) { return eIcicleError::INVALID_ARGUMENT; }

  constexpr size_t CHUNK_SIZE = 1 << 16; // 65,536 elements – maximum supported by backend per call
  *output = true;

  for (size_t offset = 0; offset < size; offset += CHUNK_SIZE) {
    size_t chunk_size = std::min(CHUNK_SIZE, size - offset);
    bool chunk_ok = true;
    eIcicleError err = check_norm_bound(input + offset, chunk_size, eNormType::LInfinity, norm_bound, cfg, &chunk_ok);
    if (err != eIcicleError::SUCCESS) { return err; }
    if (!chunk_ok) {
      *output = false;
      return eIcicleError::SUCCESS; // Early exit – entire vector fails the bound check
    }
  }
  // All chunks within bound
  return eIcicleError::SUCCESS;
}

// TODO: maybe make BaseProof more consistent by making everything Rq, since we have to convert z_hat to Rq before norm
// check anyway
bool LabradorBaseVerifier::_verify_base_proof(const LabradorBaseCaseProof& base_proof) const
{
  size_t n = lab_inst.param.n;
  size_t r = lab_inst.param.r;
  size_t d = Rq::d;

  auto& z_hat = base_proof.z_hat;
  auto& t_tilde = base_proof.t;
  auto& g_tilde = base_proof.g;
  auto& h_tilde = base_proof.h;
  auto& challenges_hat = trs.challenges_hat;
  auto final_const = lab_inst.equality_constraints[0];

  bool t_tilde_small = true, g_tilde_small = true, h_tilde_small = true;
  size_t base1 = lab_inst.param.base1;
  size_t base2 = lab_inst.param.base2;
  size_t base3 = lab_inst.param.base3;

  // 1. LInfinity checks: check t_tilde, g_tilde, h_tilde are small- correctly decomposed
  // bounds + 1 because norm check uses strict inequality
  ICICLE_CHECK(check_norm_bound_LInfinity_chunked(
    reinterpret_cast<const Zq*>(t_tilde.data()), t_tilde.size() * d, (base1 + 1) / 2 + 1, {}, &t_tilde_small));
  ICICLE_CHECK(check_norm_bound_LInfinity_chunked(
    reinterpret_cast<const Zq*>(h_tilde.data()), h_tilde.size() * d, (base2 + 1) / 2 + 1, {}, &h_tilde_small));
  ICICLE_CHECK(check_norm_bound_LInfinity_chunked(
    reinterpret_cast<const Zq*>(g_tilde.data()), g_tilde.size() * d, (base3 + 1) / 2 + 1, {}, &g_tilde_small));

  // Fail if any of the LInfinity are large
  if (!(t_tilde_small && h_tilde_small && g_tilde_small)) {
    std::cout << "LInfinity norm check failed\n";
    return false;
  }

  // 2. L2 checks

  // TODO: LInfinity for t,g,h already checked. Do we need to do a L2 check for them too?
  bool z_small = true;
  // z = INTT(z_hat)
  std::vector<Rq> z(z_hat.size());
  ICICLE_CHECK(ntt(z_hat.data(), z_hat.size(), NTTDir::kInverse, {}, z.data()));

  uint64_t op_norm_bound = lab_inst.param.op_norm_bound;
  double beta = lab_inst.param.beta;
  // Check ||z|| < op_norm*beta*sqrt(r)
  // NOTE: if n > 2^10 then this fails-- Verifier can again test this using a JL projection
  ICICLE_CHECK(check_norm_bound(
    reinterpret_cast<Zq*>(z.data()), z.size() * d, eNormType::L2, op_norm_bound * beta * sqrt(r), {}, &z_small));

  if (!z_small) {
    std::cout << "L2 norm check for z failed\n";
    return false;
  }

  // 3. Check u1, u2 commitment openings

  // compute NTTs of t_tilde, g_tilde, h_tilde
  std::vector<Tq> t_tilde_hat(t_tilde.size()), g_tilde_hat(g_tilde.size()), h_tilde_hat(h_tilde.size());
  ICICLE_CHECK(ntt(t_tilde.data(), t_tilde.size(), NTTDir::kForward, {}, t_tilde_hat.data()));
  ICICLE_CHECK(ntt(g_tilde.data(), g_tilde.size(), NTTDir::kForward, {}, g_tilde_hat.data()));
  ICICLE_CHECK(ntt(h_tilde.data(), h_tilde.size(), NTTDir::kForward, {}, h_tilde_hat.data()));

  size_t kappa1 = lab_inst.param.kappa1;

  const std::vector<Tq>& A = lab_inst.param.A;
  const std::vector<Tq>& B = lab_inst.param.B;
  const std::vector<Tq>& C = lab_inst.param.C;
  const std::vector<Tq>& D = lab_inst.param.D;
  // v1 = B@T_tilde
  std::vector<Tq> v1 = ajtai_commitment(B, t_tilde_hat.size(), kappa1, t_tilde_hat.data(), t_tilde_hat.size());
  // v2 = C@g_tilde
  std::vector<Tq> v2 = ajtai_commitment(C, g_tilde_hat.size(), kappa1, g_tilde_hat.data(), g_tilde_hat.size());
  // u1 = v1+v2
  std::vector<Tq> u1(kappa1);
  vector_add(v1.data(), v2.data(), kappa1, {}, u1.data());

  // check t_tilde, g_tilde open u1 in trs
  if (!(poly_vec_eq(u1.data(), trs.prover_msg.u1.data(), kappa1))) {
    std::cout << "t_tilde, g_tilde don't open u1 \n";
    return false;
  }

  size_t kappa2 = lab_inst.param.kappa2;
  // u2 = D@h_tilde
  std::vector<Tq> u2 = ajtai_commitment(D, h_tilde_hat.size(), kappa2, h_tilde_hat.data(), h_tilde_hat.size());

  // check h_tilde opens to u2 in trs
  if (!(poly_vec_eq(u2.data(), trs.prover_msg.u2.data(), kappa2))) {
    std::cout << "h_tilde doesn't open u2 \n";
    return false;
  }

  // 4. Check Az = \sum_i c_i*t_i

  // Use ajtai_commitment to compute z_hat @ A
  size_t kappa = lab_inst.param.kappa;
  std::vector<Tq> zA_hat = ajtai_commitment(A, n, kappa, z_hat.data(), n);

  std::vector<Rq> t(r * kappa);
  ICICLE_CHECK(recompose(t_tilde.data(), t_tilde.size(), base1, {}, t.data(), t.size()));
  std::vector<Tq> t_hat(r * kappa), ct_hat(kappa);
  // t_hat = NTT(t)
  ICICLE_CHECK(ntt(t.data(), r * kappa, NTTDir::kForward, {}, t_hat.data()));
  // ct_hat = \sum_i c_i t_i = [c1 c2 ... cr] @ t_hat
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, t_hat.data(), r, kappa, {}, ct_hat.data()));
  // zA_hat == \sum_i c_i t_i
  if (!(poly_vec_eq(zA_hat.data(), ct_hat.data(), kappa))) {
    std::cout << "_verify_base_proof failed: zA != cT \n";
    return false;
  }

  // Compute relevant matrix, vectors for the rest of the checks

  size_t r_choose_2 = (r * (r + 1)) / 2;
  std::vector<Rq> g(r_choose_2);
  ICICLE_CHECK(recompose(g_tilde.data(), g_tilde.size(), base2, {}, g.data(), g.size()));
  std::vector<Rq> G = reconstruct_symm_matrix(g, r);

  std::vector<Tq> G_hat(r * r);
  // G_hat = NTT(G)
  ICICLE_CHECK(ntt(G.data(), r * r, NTTDir::kForward, {}, G_hat.data()));

  std::vector<Rq> h(r_choose_2);
  ICICLE_CHECK(recompose(h_tilde.data(), h_tilde.size(), base3, {}, h.data(), h.size()));
  std::vector<Rq> H = reconstruct_symm_matrix(h, r);

  std::vector<Tq> H_hat(r * r);
  // H_hat = NTT(H)
  ICICLE_CHECK(ntt(H.data(), r * r, NTTDir::kForward, {}, H_hat.data()));

  Tq ip_z_z, c_G_c, c_H_c, ip_a_G, c_Phi_z, trace_H;

  // ip_z_z = <z_hat,z_hat> - inner product of z_hat with itself
  ICICLE_CHECK(matmul(z_hat.data(), 1, n, z_hat.data(), n, 1, {}, &ip_z_z));

  // c_G_c = challenges_hat^T * G_hat * challenges_hat
  // First compute G_hat * challenges_hat
  std::vector<Tq> G_times_c(r);
  ICICLE_CHECK(matmul(G_hat.data(), r, r, challenges_hat.data(), r, 1, {}, G_times_c.data()));
  // Then compute challenges_hat^T * (G_hat * challenges_hat)
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, G_times_c.data(), r, 1, {}, &c_G_c));

  // c_H_c = challenges_hat^T * H_hat * challenges_hat
  // First compute H_hat * challenges_hat
  std::vector<Tq> H_times_c(r);
  ICICLE_CHECK(matmul(H_hat.data(), r, r, challenges_hat.data(), r, 1, {}, H_times_c.data()));
  // Then compute challenges_hat^T * (H_hat * challenges_hat)
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, H_times_c.data(), r, 1, {}, &c_H_c));

  // ip_a_G = <final_const.a, G_hat> - inner product of flattened matrices
  ICICLE_CHECK(matmul(final_const.a.data(), 1, r * r, G_hat.data(), r * r, 1, {}, &ip_a_G));

  // c_Phi_z = challenges_hat^T * final_const.phi * z_hat
  // First compute phi * z_hat
  std::vector<Tq> phi_times_z(r);
  ICICLE_CHECK(matmul(final_const.phi.data(), r, n, z_hat.data(), n, 1, {}, phi_times_z.data()));
  // Then compute challenges_hat^T * (phi * z_hat)
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, phi_times_z.data(), r, 1, {}, &c_Phi_z));

  // compute trace_H = \sum_i H_ii
  ICICLE_CHECK(compute_matrix_trace(H_hat.data(), r, &trace_H));

  // c = challenges
  // 5. Ensure: <z,z> == c^t G c
  if (!(poly_vec_eq(&ip_z_z, &c_G_c, 1))) {
    std::cout << "_verify_base_proof failed: <z,z> != c^t G c \n";
    return false;
  }
  // 6. Ensure: c^t Phi z == c^t H c
  if (!(poly_vec_eq(&c_Phi_z, &c_H_c, 1))) {
    std::cout << "_verify_base_proof failed: c^t Phi z != c^t H c\n";
    return false;
  }

  // 7. Ensure: \sum_ij a_ij G_ij + \sum_i h_ii + b == 0
  // \sum_ij a_ij G_ij + \sum_i h_ii
  Tq ip_a_G_plus_trace_H;
  ICICLE_CHECK(vector_add(&ip_a_G, &trace_H, 1, {}, &ip_a_G_plus_trace_H));

  Tq ip_a_G_plus_trace_H_plus_b;
  ICICLE_CHECK(vector_add(&ip_a_G_plus_trace_H, &final_const.b, 1, {}, &ip_a_G_plus_trace_H_plus_b));

  Tq zero_poly(zero());
  // Check \sum_ij a_ij G_ij + \sum_i h_ii + b == 0
  if (!(poly_vec_eq(&ip_a_G_plus_trace_H_plus_b, &zero_poly, 1))) {
    std::cout << "_verify_base_proof failed: sum_ij a_ij G_ij + sum_i h_ii + b !=0\n";
    return false;
  }
  return true;
}

// modifies the instance
// Doesn't perform any checks
// returns num_aggregation_rounds number of polynomials
void LabradorBaseVerifier::agg_const_zero_constraints()
{
  size_t r = lab_inst.param.r;
  size_t n = lab_inst.param.n;
  size_t d = Rq::d;
  size_t JL_out = lab_inst.param.JL_out;
  size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;
  const size_t L = lab_inst.const_zero_constraints.size();

  size_t JL_i = trs.prover_msg.JL_i;
  const std::vector<std::byte> seed1 = trs.seed1;
  const std::vector<Zq>& p = trs.prover_msg.p;
  const std::vector<Zq>& psi = trs.psi;
  const std::vector<Zq>& omega = trs.omega;
  // indexes into multidim arrays: psi[k][l] and omega[k][l]
  auto psi_index = [num_aggregation_rounds, L](size_t k, size_t l) {
    assert(l < L);
    assert(k < num_aggregation_rounds);
    return k * L + l;
  };
  auto omega_index = [num_aggregation_rounds, JL_out](size_t k, size_t l) {
    assert(l < JL_out);
    assert(k < num_aggregation_rounds);
    return k * JL_out + l;
  };

  std::vector<std::byte> jl_seed(seed1);
  jl_seed.push_back(std::byte(JL_i));

  std::vector<Zq> test_b0(num_aggregation_rounds, Zq::zero());

  for (size_t k = 0; k < num_aggregation_rounds; k++) {
    EqualityInstance new_constraint(r, n);
    std::vector<ConstZeroInstance> temp_const(lab_inst.const_zero_constraints);

    // Compute a''_{ij} = sum_{l=0}^{L-1} psi^{(k)}(l) * a'_{ij}^{(l)}

    // For each l do:
    // const_zero_constraints[l].a[i,j] = psi[k,l]* const_zero_constraints[l].a[i,j]
    // use async_config to parallelise
    VecOpsConfig async_config = default_vec_ops_config();
    async_config.is_async = true;

    for (size_t l = 0; l < L; l++) {
      Zq psi_scalar = psi[psi_index(k, l)];

      ICICLE_CHECK(scalar_mul_vec(
        &psi_scalar, reinterpret_cast<Zq*>(temp_const[l].a.data()), r * r * d, async_config,
        reinterpret_cast<Zq*>(temp_const[l].a.data())));
    }
    ICICLE_CHECK(icicle_device_synchronize());
    // new_constraint.a[i,j] = \sum_l const_zero_constraints[l].a[i,j]
    for (size_t l = 0; l < L; l++) {
      ICICLE_CHECK(vector_add(new_constraint.a.data(), temp_const[l].a.data(), r * r, {}, new_constraint.a.data()));
    }

    // Compute varphi'_i^{(k)} = sum_{l=0}^{L-1} psi^{(k)}(l) * phi'_i^{(l)} + sum_{l=0}^{255} omega^{(k)}(l) * q_{il}

    // For each l do:
    // const_zero_constraints[l].phi[i,:] = psi[k,l]* const_zero_constraints[l].phi[i,:]
    // use async_config to parallelise
    // TODO: can async with a aggregation above- leave for later
    for (size_t l = 0; l < L; l++) {
      Zq psi_scalar = psi[psi_index(k, l)];

      ICICLE_CHECK(scalar_mul_vec(
        &psi_scalar, reinterpret_cast<Zq*>(temp_const[l].phi.data()), r * n * d, async_config,
        reinterpret_cast<Zq*>(temp_const[l].phi.data())));
    }
    ICICLE_CHECK(icicle_device_synchronize());
    // new_constraint.phi[i,:] = \sum_l const_zero_constraints[l].phi[i,:]
    for (size_t l = 0; l < L; l++) {
      ICICLE_CHECK(
        vector_add(new_constraint.phi.data(), temp_const[l].phi.data(), r * n, {}, new_constraint.phi.data()));
    }

    // For each j do:
    //    new_constraint.phi[:,:] += omega[k,j]* Q_hat[j, :, :]
    for (size_t j = 0; j < JL_out; j++) {
      Zq omega_scalar = omega[omega_index(k, j)];

      // Q_j = Q_hat[j, :, :]
      std::vector<Rq> Q_j(r * n);
      // compute the Pi matrix row, conjugated in Rq
      ICICLE_CHECK(icicle::labrador::get_jl_matrix_rows(
        jl_seed.data(), jl_seed.size(),
        r * n, // row_size
        j,     // row_index
        1,     // num_rows
        true,  // conjugate
        {},    // config
        Q_j.data()));

      std::vector<Tq> Q_j_hat(r * n);
      // Q_j_hat = NTT(Q_j)
      ICICLE_CHECK(ntt(Q_j.data(), Q_j.size(), NTTDir::kForward, {}, Q_j_hat.data()));
      // Q_hat[j, :, :] = omega[k,j]* Q_hat[j, :, :]
      ICICLE_CHECK(scalar_mul_vec(
        &omega_scalar, reinterpret_cast<Zq*>(Q_j_hat.data()), r * n * d, {}, reinterpret_cast<Zq*>(Q_j_hat.data())));

      ICICLE_CHECK(vector_add(new_constraint.phi.data(), Q_j_hat.data(), r * n, {}, new_constraint.phi.data()));
    }

    new_constraint.b = trs.prover_msg.b_agg[k];

    // Add the EqualityInstance to LabradorInstance
    lab_inst.add_equality_constraint(new_constraint);
  }
  // delete the const zero constraints
  lab_inst.const_zero_constraints.clear();
  lab_inst.const_zero_constraints.shrink_to_fit();
}

// Verifies transcript messages are valid
// Also aggregates the lab_inst into the correct final constraint
bool LabradorBaseVerifier::part_verify()
{
  size_t r = lab_inst.param.r;
  size_t n = lab_inst.param.n;
  size_t d = Rq::d;
  size_t JL_out = lab_inst.param.JL_out;
  size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;

  const std::vector<Zq>& p = trs.prover_msg.p;
  const std::vector<Tq>& b_agg = trs.prover_msg.b_agg;
  const std::vector<Zq>& psi = trs.psi;
  const std::vector<Zq>& omega = trs.omega;

  std::vector<Rq> b_agg_unhat(b_agg.size());
  // TODO: don't need complete NTT on Verifier side
  ICICLE_CHECK(ntt(b_agg.data(), b_agg.size(), NTTDir::kInverse, {}, b_agg_unhat.data()));

  // create_transcript called in constructor - so transcript is ready to be used

  // check p norm
  bool JL_check = false;
  double beta = lab_inst.param.beta;
  ICICLE_CHECK(check_norm_bound(p.data(), JL_out, eNormType::L2, uint64_t(sqrt(JL_out / 2) * beta), {}, &JL_check));
  if (!JL_check) {
    std::cout << "verify(): p-norm check failed" << std::endl;
    return false;
  }

  // b_agg have correct const term
  std::vector<Zq> test_b0(num_aggregation_rounds, Zq::zero());
  const size_t L = lab_inst.const_zero_constraints.size();

  auto psi_index = [num_aggregation_rounds, L](size_t k, size_t l) {
    assert(l < L);
    assert(k < num_aggregation_rounds);
    return k * L + l;
  };
  auto omega_index = [num_aggregation_rounds, JL_out](size_t k, size_t l) {
    assert(l < JL_out);
    assert(k < num_aggregation_rounds);
    return k * JL_out + l;
  };
  for (size_t k = 0; k < num_aggregation_rounds; k++) {
    for (size_t l = 0; l < L; l++) {
      test_b0[k] = test_b0[k] + psi[psi_index(k, l)] * lab_inst.const_zero_constraints[l].b;
    }
    for (size_t l = 0; l < JL_out; l++) {
      test_b0[k] = test_b0[k] - omega[omega_index(k, l)] * p[l];
    }

    // Zq lhs = b_agg_unhat[k].values[0];
    // Zq rhs = test_b0[k]; // already available
    // Zq diff = lhs - rhs; // mod q

    // std::cout << "k=" << k << "  lhs=" << lhs << "  rhs=" << rhs << "  diff=" << diff << std::endl;

    if (test_b0[k] != b_agg_unhat[k].values[0]) {
      std::cout << "verify(): b0 test failed for " << k << std::endl;
      return false;
    }
  }

  // construct the final constraint correctly
  agg_const_zero_constraints();
  lab_inst.agg_equality_constraints(trs.alpha_hat);

  return true;
}

bool LabradorBaseVerifier::fully_verify(const LabradorBaseCaseProof& base_proof)
{
  if (part_verify()) {
    return _verify_base_proof(base_proof);
  } else {
    return false;
  }
}

bool LabradorVerifier::verify()
{
  LabradorInstance lab_inst_i = lab_inst;
  for (size_t i = 0; i < NUM_REC - 1; i++) {
    std::cout << "Verifier::Recursion iteration = " << i << "\n";
    LabradorBaseVerifier base_verifier(lab_inst_i, prover_msgs[i], oracle);
    if (!base_verifier.part_verify()) {
      std::cout << "\tProver message verification failed\n";
      return false;
    }
    // Part verify correctly aggregates constraints

    // Prepare recursion problem
    // NOTE: base0 needs to be large enough
    uint32_t base0 = calc_base0(lab_inst_i.param.r, OP_NORM_BOUND, lab_inst_i.param.beta);
    size_t m = lab_inst_i.param.t_len() + lab_inst_i.param.g_len() + lab_inst_i.param.h_len();
    auto [mu, nu] = compute_mu_nu(lab_inst_i.param.n, m);

    EqualityInstance final_const = base_verifier.lab_inst.equality_constraints[0];
    lab_inst_i =
      prepare_recursion_instance(base_verifier.lab_inst.param, final_const, base_verifier.trs, base0, mu, nu);
    oracle = base_verifier.oracle;

    std::cout << "\tVerifier::Recursion problem prepared\n";
    std::cout << "\tn= " << lab_inst_i.param.n << ", r= " << lab_inst_i.param.r << "\n";
  }
  std::cout << "Verifier::Recursion iteration = " << NUM_REC - 1 << "\n";
  LabradorBaseVerifier base_verifier(lab_inst_i, prover_msgs[NUM_REC - 1], oracle);
  if (!base_verifier.fully_verify(final_proof)) {
    std::cout << "\tVerifier- Final verification failed\n";
    return false;
  }
  return true;
}
