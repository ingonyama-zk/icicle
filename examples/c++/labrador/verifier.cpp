#include "verifier.h"

// TODO: maybe make BaseProof more consistent by making everything Rq, since we have to convert z_hat to Rq before norm
// check anyway
bool LabradorBaseVerifier::_verify_base_proof() const
{
  size_t n = lab_inst.param.n;
  size_t r = lab_inst.param.r;
  size_t d = Rq::d;

  auto z_hat = base_proof.z_hat;
  auto t_tilde = base_proof.t;
  auto g_tilde = base_proof.g;
  auto h_tilde = base_proof.h;
  auto challenges_hat = trs.challenges_hat;

  bool t_tilde_small = true, g_tilde_small = true, h_tilde_small = true;
  size_t base1 = lab_inst.param.base1;
  size_t base2 = lab_inst.param.base2;
  size_t base3 = lab_inst.param.base3;

  // 1. LInfinity checks: check t_tilde, g_tilde, h_tilde are small- correctly decomposed

  ICICLE_CHECK(check_norm_bound(
    reinterpret_cast<Zq*>(t_tilde.data()), t_tilde.size() * d, eNormType::LInfinity, (base1 + 1) / 2, {},
    &t_tilde_small));
  ICICLE_CHECK(check_norm_bound(
    reinterpret_cast<Zq*>(h_tilde.data()), h_tilde.size() * d, eNormType::LInfinity, (base2 + 1) / 2, {},
    &h_tilde_small));
  ICICLE_CHECK(check_norm_bound(
    reinterpret_cast<Zq*>(g_tilde.data()), g_tilde.size() * d, eNormType::LInfinity, (base3 + 1) / 2, {},
    &g_tilde_small));

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

  const std::vector<std::byte>& ajtai_seed = lab_inst.param.ajtai_seed;
  std::vector<std::byte> seed_A(ajtai_seed), seed_B(ajtai_seed), seed_C(ajtai_seed), seed_D(ajtai_seed);
  seed_A.push_back(std::byte('0'));
  seed_B.push_back(std::byte('1'));
  seed_C.push_back(std::byte('2'));
  seed_D.push_back(std::byte('3'));

  size_t kappa1 = lab_inst.param.kappa1;
  // v1 = B@T_tilde
  std::vector<Tq> v1 =
    ajtai_commitment(seed_B.data(), seed_B.size(), t_tilde_hat.size(), kappa1, t_tilde_hat.data(), t_tilde_hat.size());
  // v2 = C@g_tilde
  std::vector<Tq> v2 =
    ajtai_commitment(seed_C.data(), seed_C.size(), g_tilde_hat.size(), kappa1, g_tilde_hat.data(), g_tilde_hat.size());
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
  std::vector<Tq> u2 =
    ajtai_commitment(seed_D.data(), seed_D.size(), h_tilde_hat.size(), kappa2, h_tilde_hat.data(), h_tilde_hat.size());

  // check h_tilde opens to u2 in trs
  if (!(poly_vec_eq(u2.data(), trs.prover_msg.u2.data(), kappa2))) {
    std::cout << "h_tilde doesn't open u2 \n";
    return false;
  }

  // 4. Check Az = \sum_i c_i*t_i

  // Use ajtai_commitment to compute z_hat @ A
  size_t kappa = lab_inst.param.kappa;
  std::vector<Tq> zA_hat = ajtai_commitment(seed_A.data(), seed_A.size(), n, kappa, z_hat.data(), n);

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

  // ip_a_G = <base_proof.final_const.a, G_hat> - inner product of flattened matrices
  ICICLE_CHECK(matmul(base_proof.final_const.a.data(), 1, r * r, G_hat.data(), r * r, 1, {}, &ip_a_G));

  // c_Phi_z = challenges_hat^T * base_proof.final_const.phi * z_hat
  // First compute phi * z_hat
  std::vector<Tq> phi_times_z(r);
  ICICLE_CHECK(matmul(base_proof.final_const.phi.data(), r, n, z_hat.data(), n, 1, {}, phi_times_z.data()));
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
  ICICLE_CHECK(vector_add(&ip_a_G_plus_trace_H, &base_proof.final_const.b, 1, {}, &ip_a_G_plus_trace_H_plus_b));

  Tq zero_poly(zero());
  // Check \sum_ij a_ij G_ij + \sum_i h_ii + b == 0
  if (!(poly_vec_eq(&ip_a_G_plus_trace_H_plus_b, &zero_poly, 1))) {
    std::cout << "_verify_base_proof failed: sum_ij a_ij G_ij + sum_i h_ii + b !=0\n";
    return false;
  }
  return true;
}

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
  const size_t num_agg_rounds = std::ceil(128.0 / std::log2(get_q<Zq>()));
  trs.psi.resize(num_agg_rounds * L);
  trs.omega.resize(num_agg_rounds * JL_out);
  // psi seed = seed2 || 0x01
  std::vector<std::byte> psi_seed(trs.seed2);
  psi_seed.push_back(std::byte('1'));
  random_sampling(psi_seed.data(), psi_seed.size(), false, {}, trs.psi.data(), trs.psi.size());
  // omega seed = seed2 || 0x02
  std::vector<std::byte> omega_seed(trs.seed2);
  omega_seed.push_back(std::byte('2'));
  random_sampling(omega_seed.data(), omega_seed.size(), false, {}, trs.omega.data(), trs.omega.size());

  // 4. seed3 from msg3 (b_agg)
  const auto& msg3 = trs.prover_msg.b_agg;
  const std::byte* msg3_bytes = reinterpret_cast<const std::byte*>(msg3.data());
  size_t msg3_len = msg3.size() * sizeof(Tq);
  trs.seed3 = oracle.generate(msg3_bytes, msg3_len);

  // 5. alpha_hat sampling
  const size_t K = lab_inst.equality_constraints.size();
  trs.alpha_hat.resize(K);
  std::vector<std::byte> alpha_seed(trs.seed3);
  alpha_seed.push_back(std::byte('1'));
  random_sampling(alpha_seed.data(), alpha_seed.size(), false, {}, trs.alpha_hat.data(), K);

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

bool LabradorBaseVerifier::verify() const { return true; }