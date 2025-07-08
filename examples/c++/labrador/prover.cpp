#include "prover.h"
#include <cassert>

std::pair<size_t, std::vector<Zq>> LabradorBaseProver::select_valid_jl_proj(std::byte* seed, size_t seed_len) const
{
  size_t JL_out = lab_inst.param.JL_out;
  size_t n = lab_inst.param.n;
  size_t r = lab_inst.param.r;
  size_t d = Rq::d;

  std::vector<Zq> p(JL_out);
  size_t JL_i = 0;
  std::vector<std::byte> jl_seed(seed, seed + seed_len);
  while (true) {
    jl_seed.push_back(std::byte(JL_i));
    // create JL projection: P*(s_1, s_2, ..., s_r)
    ICICLE_CHECK(icicle::labrador::jl_projection(
      reinterpret_cast<const Zq*>(S.data()), n * r * d, jl_seed.data(), jl_seed.size(), {}, p.data(), JL_out));
    // check norm
    bool JL_check = false;
    double beta = lab_inst.param.beta;

    // ignore ICICLE errors when elements of p are greater than sqrt(q)
    try {
      ICICLE_CHECK(check_norm_bound(p.data(), JL_out, eNormType::L2, uint64_t(sqrt(JL_out / 2) * beta), {}, &JL_check));
    } catch (const std::exception& e) {
      // simply pass here
    }

    if (JL_check) {
      break;
    } else {
      p.assign(p.size(), Zq::from(0));
      JL_i++;
      jl_seed.pop_back();
    }
  }
  // at the end JL projection is defined by JL_i and p is the projection output
  // return these
  return std::make_pair(JL_i, p);
}

// modifies the instance
// returns num_aggregation_rounds number of polynomials
std::vector<Tq> LabradorBaseProver::agg_const_zero_constraints(
  const std::vector<Tq>& S_hat,
  const std::vector<Tq>& G_hat,
  const std::vector<Zq>& p,
  const std::vector<Zq>& psi,
  const std::vector<Zq>& omega,
  size_t JL_i,
  const std::vector<std::byte>& seed1)
{
  size_t r = lab_inst.param.r;
  size_t n = lab_inst.param.n;
  size_t d = Rq::d;
  size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;
  size_t JL_out = lab_inst.param.JL_out;
  const size_t L = lab_inst.const_zero_constraints.size();

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

  std::vector<Zq> verif_test_b0(num_aggregation_rounds, Zq::zero());

  std::vector<Tq> msg3;
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

      // Make sure that <Q_j_hat, S_hat> - p_j == 0
      if (TESTING) {
        ConstZeroInstance cz{r, n};
        cz.phi = Q_j_hat;
        cz.b = Zq::neg(p[j]);
        if (!witness_legit_const_zero_all_ntt(cz, S_hat)) {
          std::cout << "Q_j eqn error for j = " << j << "\n";
          exit(1);
        }
      }

      // Q_hat[j, :, :] = omega[k,j]* Q_hat[j, :, :]
      ICICLE_CHECK(scalar_mul_vec(
        &omega_scalar, reinterpret_cast<Zq*>(Q_j_hat.data()), r * n * d, {}, reinterpret_cast<Zq*>(Q_j_hat.data())));

      ICICLE_CHECK(vector_add(new_constraint.phi.data(), Q_j_hat.data(), r * n, {}, new_constraint.phi.data()));
    }

    // Compute B^{(k)} = sum_{ij} a''_{ij}^{(k)}  * g_{ij} + sum_i <phi'_i^{(k)}, s_i>
    Tq G_A_inner_prod, phi_S_inner_prod;
    // G_A_inner_prod = <G, a>
    ICICLE_CHECK(matmul(G_hat.data(), 1, r * r, new_constraint.a.data(), r * r, 1, {}, &G_A_inner_prod));
    // phi_S_inner_prod = <S, phi>
    ICICLE_CHECK(matmul(S_hat.data(), 1, r * n, new_constraint.phi.data(), r * n, 1, {}, &phi_S_inner_prod));
    // b = -(<G, a> + <S, phi>)
    ICICLE_CHECK(vector_add(G_A_inner_prod.values, phi_S_inner_prod.values, d, {}, new_constraint.b.values));
    Zq minus_1 = Zq::neg(Zq::from(1));
    ICICLE_CHECK(scalar_mul_vec(&minus_1, new_constraint.b.values, d, {}, new_constraint.b.values));

    // TODO: is this needed
    ICICLE_CHECK(icicle_device_synchronize());

    if (TESTING) {
      // Following should work if our B^{(k)} evaluation is correct above
      if (!witness_legit_eq(new_constraint, S)) { std::cout << "Constraint " << k << " failed\n"; }

      // Verifier performs these checks
      for (size_t l = 0; l < L; l++) {
        verif_test_b0[k] = verif_test_b0[k] + psi[psi_index(k, l)] * lab_inst.const_zero_constraints[l].b;
      }
      for (size_t l = 0; l < JL_out; l++) {
        verif_test_b0[k] = verif_test_b0[k] - omega[omega_index(k, l)] * p[l];
      }

      Rq b_rq;
      ICICLE_CHECK(ntt(&new_constraint.b, 1, NTTDir::kInverse, {}, &b_rq));

      Zq lhs = b_rq.values[0];
      Zq rhs = verif_test_b0[k]; // already available
      Zq diff = lhs - rhs;       // mod q

      std::cout << "k=" << k << "  lhs=" << lhs << "  rhs=" << rhs << "  diff=" << diff << std::endl;
      if (verif_test_b0[k] != b_rq.values[0]) {
        std::cout << "\tFail: New constraint b doesn't match verif b for idx" << k << "\n";
      } else {
        std::cout << "\tPass: New constraint b matches verif b for idx" << k << "\n";
      }
      // if (!witness_legit_const_zero({r, n, new_constraint.a, new_constraint.phi, verif_test_b0[k]}, S)) {
      //   std::cout << "\tVerif test constraint " << k << " failed\n";
      // }
    }
    // Add the EqualityInstance to LabradorInstance
    lab_inst.add_equality_constraint(new_constraint);

    // Send B^(k) to the Verifier
    msg3.push_back(new_constraint.b);
  }

  // delete the const zero constraints
  lab_inst.const_zero_constraints.clear();
  lab_inst.const_zero_constraints.shrink_to_fit();

  return msg3;
}

// This destroys the lab_inst in LabradorBaseProver
// TODO: Prover only needs to send BaseProverMessages
std::pair<LabradorBaseCaseProof, PartialTranscript> LabradorBaseProver::base_case_prover()
{
  // Step 1: Pack the Witnesses into a Matrix S
  const size_t r = lab_inst.param.r; // Number of witness vectors
  const size_t n = lab_inst.param.n; // Dimension of witness vectors
  constexpr size_t d = Rq::d;

  std::cout << "Number of constraints (Eq, Cz): " << lab_inst.equality_constraints.size() << ", "
            << lab_inst.const_zero_constraints.size() << std::endl;

  PartialTranscript trs;
  std::cout << "Step 1 completed: Initialized variables" << std::endl;

  // Step 2: Convert S to the NTT Domain
  std::vector<Tq> S_hat(r * n);
  // Perform negacyclic NTT on the witness S
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));
  std::cout << "Step 2 completed: NTT conversion" << std::endl;

  // Step 3: S@A = T
  size_t kappa = lab_inst.param.kappa;
  const std::vector<Tq>& A = lab_inst.param.A; // n × kappa matrix

  // Compute T_hat = S_hat @ A
  std::vector<Tq> T_hat = ajtai_commitment(A, n, kappa, S_hat.data(), r * n);
  std::cout << "Step 3 completed: Ajtai commitment T_hat" << std::endl;

  // Step 4: already done

  // Step 5: Convert T_hat to Rq
  std::vector<Rq> T(r * kappa);
  // Perform negacyclic INTT
  ICICLE_CHECK(ntt(T_hat.data(), r * kappa, NTTDir::kInverse, {}, T.data()));
  std::cout << "Step 5 completed: INTT conversion of T_hat" << std::endl;

  // Step 6: decompose T to T_tilde
  size_t base1 = lab_inst.param.base1;
  size_t l1 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base1);
  std::vector<Rq> T_tilde(l1 * r * kappa);
  ICICLE_CHECK(decompose(T.data(), r * kappa, base1, {}, T_tilde.data(), T_tilde.size()));
  std::cout << "Step 6 completed: Decomposed T to T_tilde" << std::endl;

  if (TESTING) {
    // Ensure that recompose(T_tilde) == T
    std::vector<Rq> temp(r * kappa);
    ICICLE_CHECK(recompose(T_tilde.data(), T_tilde.size(), base1, {}, temp.data(), temp.size()));
    bool decompose_recompose_correct = true;
    for (size_t i = 0; i < r * kappa; i++) {
      for (size_t j = 0; j < d; j++) {
        if (temp[i].values[j] != T[i].values[j]) {
          decompose_recompose_correct = false;
          break;
        }
      }
      if (!decompose_recompose_correct) break;
    }
    if (decompose_recompose_correct) {
      std::cout << "\tDecompose/recompose test passed\n";
    } else {
      std::cout << "\tDecompose/recompose test failed\n";
    }
  }

  // Step 7: compute g
  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  std::vector<Tq> g_hat = extract_symm_part(G_hat.data(), r);
  size_t r_choose_2 = (r * (r + 1)) / 2;
  std::vector<Rq> g(r_choose_2);

  ICICLE_CHECK(ntt(g_hat.data(), r_choose_2, NTTDir::kInverse, {}, g.data()));
  std::cout << "Step 7 completed: Computed g" << std::endl;

  // Step 8: decompose g to g_tilde
  size_t base2 = lab_inst.param.base2;
  // TODO: Take advantage of g being small and truncate upper limbs
  size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base2);
  std::vector<Rq> g_tilde(l2 * g.size());
  ICICLE_CHECK(decompose(g.data(), g.size(), base2, {}, g_tilde.data(), g_tilde.size()));
  std::cout << "Step 8 completed: Decomposed g to g_tilde" << std::endl;

  // Step 9: already done

  // Step 10: u1 = B@T_tilde + C@g_tilde
  size_t kappa1 = lab_inst.param.kappa1;
  const std::vector<Tq>& B = lab_inst.param.B; // (t_len × kappa1)
  const std::vector<Tq>& C = lab_inst.param.C; // (g_len × kappa1)

  // compute NTTs for T_tilde, g_tilde
  std::vector<Tq> T_tilde_hat(T_tilde.size()), g_tilde_hat(g_tilde.size());
  ICICLE_CHECK(ntt(T_tilde.data(), T_tilde.size(), NTTDir::kForward, {}, T_tilde_hat.data()));
  ICICLE_CHECK(ntt(g_tilde.data(), g_tilde.size(), NTTDir::kForward, {}, g_tilde_hat.data()));

  // v1 = B @ T_tilde
  std::vector<Tq> v1 = ajtai_commitment(B, T_tilde_hat.size(), kappa1, T_tilde_hat.data(), T_tilde_hat.size());
  // v2 = C @ g_tilde
  std::vector<Tq> v2 = ajtai_commitment(C, g_tilde_hat.size(), kappa1, g_tilde_hat.data(), g_tilde_hat.size());

  std::vector<Tq> u1(kappa1);
  vector_add(v1.data(), v2.data(), kappa1, {}, u1.data());
  std::cout << "Step 10 completed: Computed u1" << std::endl;

  // Step 11: derive seed1 using the oracle and the actual bytes of u1
  const std::byte* u1_bytes = reinterpret_cast<const std::byte*>(u1.data());
  const size_t u1_bytes_len = u1.size() * sizeof(Tq);
  std::vector<std::byte> seed1 = oracle.generate(u1_bytes, u1_bytes_len);
  // add u1 to the trs
  trs.prover_msg.u1 = u1;
  trs.seed1 = seed1;
  std::cout << "Step 11 completed: Generated seed1" << std::endl;

  // Step 12: Select a JL projection
  size_t JL_out = lab_inst.param.JL_out;
  auto [JL_i, p] = select_valid_jl_proj(seed1.data(), seed1.size());
  std::cout << "Step 12 completed: Selected JL projection" << std::endl;

  trs.prover_msg.JL_i = JL_i;
  trs.prover_msg.p = p;

  // Step 13: serialize (JL_i, p) into bytes and feed to oracle for seed2
  std::vector<std::byte> jl_buf(sizeof(size_t));
  std::memcpy(jl_buf.data(), &JL_i, sizeof(size_t));
  jl_buf.insert(
    jl_buf.end(), reinterpret_cast<const std::byte*>(p.data()),
    reinterpret_cast<const std::byte*>(p.data()) + p.size() * sizeof(Zq));

  std::vector<std::byte> seed2 = oracle.generate(jl_buf.data(), jl_buf.size());
  trs.seed2 = seed2;
  std::cout << "Step 13 completed: Generated seed2" << std::endl;

  // Step 14: removed
  // Step 15, 16: already done

  // Step 17: Create conjugated polynomial vectors from JL matrix rows
  std::cout << "Step 17 completed: Computed Q polynomial" << std::endl;

  // Step 18: Let L be the number of constZeroInstance constraints in LabradorInstance.
  // For 0 ≤ k < ceil(128/log(q)), sample the following random vectors:
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;

  std::vector<Zq> psi(num_aggregation_rounds * L), omega(num_aggregation_rounds * JL_out);

  // sample psi
  // psi seed = seed2 || 0x01
  std::vector<std::byte> psi_seed(seed2);
  psi_seed.push_back(std::byte('1'));
  // TODO: change fast mode to false
  if (psi.size() > 0) {
    ICICLE_CHECK(random_sampling(psi.size(), true, psi_seed.data(), psi_seed.size(), {}, psi.data()));
  }
  // Sample omega
  // omega seed = seed2 || 0x02
  std::vector<std::byte> omega_seed(seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(omega.size(), true, omega_seed.data(), omega_seed.size(), {}, omega.data()));

  trs.psi = psi;
  trs.omega = omega;
  std::cout << "Step 18 completed: Sampled psi and omega" << std::endl;

  // Step 19: Aggregate ConstZeroInstance constraints

  // if (TESTING) {
  //   bool Q_testing = true;
  //   for (size_t l = 0; l < JL_out; l++) {
  //     std::vector<Tq> a{r * r, zero()}, phi{&Q_hat[l * r * n], &Q_hat[(l + 1) * r * n]};

  //     ConstZeroInstance cz{r, n, a, phi, Zq::neg(p[l])};
  //     if (!witness_legit_const_zero(cz, S)) {
  //       std::cout << "\tQ-constraint-check fails for " << l << "\n";
  //       Q_testing = false;
  //       break;
  //     };
  //   }
  //   if (Q_testing) { std::cout << "\tQ-constraint-check passed... " << "\n"; }
  // }

  std::vector<Tq> msg3 = agg_const_zero_constraints(S_hat, G_hat, p, psi, omega, JL_i, seed1);
  std::cout << "Step 19 completed: Aggregated ConstZeroInstance constraints" << std::endl;

  if (TESTING) {
    std::cout << "\tTesting witness validity...";
    assert(lab_witness_legit(lab_inst, S));
    std::cout << "VALID\n";
  }

  // Step 20: derive seed3 from the oracle using the actual bytes of msg3
  const std::byte* msg3_bytes = reinterpret_cast<const std::byte*>(msg3.data());
  const size_t msg3_bytes_len = msg3.size() * sizeof(Tq);
  std::vector<std::byte> seed3 = oracle.generate(msg3_bytes, msg3_bytes_len);

  trs.prover_msg.b_agg = msg3;
  trs.seed3 = seed3;
  std::cout << "Step 20 completed: Generated seed3" << std::endl;

  // Step 21: Sample random polynomial vectors α using seed3
  // Let K be the number of EqualityInstances in the LabradorInstance
  const size_t K = lab_inst.equality_constraints.size();
  assert(K > 0);
  std::vector<Tq> alpha_hat(K);
  std::vector<std::byte> alpha_seed(seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(K, true, alpha_seed.data(), alpha_seed.size(), {}, alpha_hat.data()));

  trs.alpha_hat = alpha_hat;
  std::cout << "Step 21 completed: Sampled alpha_hat" << std::endl;

  // Step 22:
  lab_inst.agg_equality_constraints(alpha_hat);
  std::cout << "Step 22 completed: Aggregated equality constraints" << std::endl;
  if (TESTING) {
    std::cout << "\tTesting witness validity...";
    assert(lab_witness_legit(lab_inst, S));
    std::cout << "VALID\n";
  }

  // Step 23: For 0 ≤ i ≤ j < r, the Prover computes the matrix multiplication between matrix
  // Phi = (φ'_0|φ'_1|···|φ'_{r-1})^T ∈ R_q^{r×n} and S ∈ R_q^{r×n} defined earlier.
  // Let H ∈ R_q^{r×r}, such that H = 2^{-1}(Phi @ S^T + (Phi @ S^T)^T)

  // Matrix Phi
  const Tq* phi_final = lab_inst.equality_constraints[0].phi.data();

  // Compute Phi @ S^T using the transposed S_hat
  std::vector<Tq> Phi_times_St_hat(r * r);
  ICICLE_CHECK(matmul(phi_final, r, n, S_hat_transposed.data(), n, r, {}, Phi_times_St_hat.data()));

  // Convert back to Rq domain
  std::vector<Rq> H(r * r), Phi_times_St_transposed(r * r);
  // H = Phi @ S^t
  ICICLE_CHECK(ntt(Phi_times_St_hat.data(), r * r, NTTDir::kInverse, {}, H.data()));
  // transpose matrix
  ICICLE_CHECK(matrix_transpose<Tq>(H.data(), r, r, {}, Phi_times_St_transposed.data()));

  // Compute H = 2^{-1}(LS + (LS)^T)
  Zq two_inv = Zq::inverse(Zq::from(2)); // 2^{-1} in Z_q

  // H = H + Phi_times_St_transposed = Phi@S^t + Phi_times_St_transposed
  ICICLE_CHECK(vector_add(H.data(), Phi_times_St_transposed.data(), r * r, {}, H.data()));
  // H = 1/2 * H
  ICICLE_CHECK(
    scalar_mul_vec(&two_inv, reinterpret_cast<Zq*>(H.data()), r * r * d, {}, reinterpret_cast<Zq*>(H.data())));

  std::vector<Rq> h = extract_symm_part(H.data(), r);
  std::cout << "Step 23 completed: Computed h vector" << std::endl;

  // Step 24: Decompose h
  size_t base3 = lab_inst.param.base3;
  size_t l3 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base3);

  std::vector<Rq> h_tilde(l3 * h.size());
  ICICLE_CHECK(decompose(h.data(), h.size(), base3, {}, h_tilde.data(), h_tilde.size()));
  std::vector<Tq> h_tilde_hat(h_tilde.size());
  ICICLE_CHECK(ntt(h_tilde.data(), h_tilde.size(), NTTDir::kForward, {}, h_tilde_hat.data()));
  std::cout << "Step 24 completed: Decomposed h to H_tilde" << std::endl;

  // Step 25: already done
  // Step 26: commit to h_tilde
  size_t kappa2 = lab_inst.param.kappa2;
  const std::vector<Tq>& D = lab_inst.param.D; // (h_len × kappa2)

  // u2 = D @ h_tilde
  std::vector<Tq> u2 = ajtai_commitment(D, h_tilde_hat.size(), kappa2, h_tilde_hat.data(), h_tilde_hat.size());
  std::cout << "Step 26 completed: Computed u2 commitment" << std::endl;

  // Step 27:
  // add u2 to the trs
  trs.prover_msg.u2 = u2;

  // Derive seed4 from oracle using bytes of u2
  const std::byte* u2_bytes = reinterpret_cast<const std::byte*>(u2.data());
  const size_t u2_bytes_len = u2.size() * sizeof(Tq);
  std::vector<std::byte> seed4 = oracle.generate(u2_bytes, u2_bytes_len);

  trs.seed4 = seed4;
  std::cout << "Step 27 completed: Generated seed4" << std::endl;

  // Step 28: sampling low operator norm challenges
  std::vector<Rq> challenge = sample_low_norm_challenges(n, r, seed4.data(), seed4.size());

  std::vector<Tq> challenges_hat(r);
  ICICLE_CHECK(ntt(challenge.data(), challenge.size(), NTTDir::kForward, {}, challenges_hat.data()));
  trs.challenges_hat = challenges_hat;
  std::cout << "Step 28 completed: Sampled challenges" << std::endl;

  // Step 29: Compute z_hat[:] = \sum_i c_i * S[i,:] = [c1 c2 ... cr] @ S
  std::vector<Tq> z_hat(n);
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, S_hat.data(), r, n, {}, z_hat.data()));

  if (TESTING) {
    std::vector<Tq> ct_hat(kappa);
    ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, T_hat.data(), r, kappa, {}, ct_hat.data()));
    std::vector<Tq> zA_hat = ajtai_commitment(A, n, kappa, z_hat.data(), z_hat.size());

    // zA_hat == \sum_i c_i t_i
    bool succ = true;
    for (size_t i = 0; i < kappa; i++) {
      for (size_t j = 0; j < d; j++) {
        if (zA_hat[i].values[j] != ct_hat[i].values[j]) {
          succ = false;
          std::cout << "\tbase_prover zA = ct failed\n";
          break;
        }
      }
    }
    if (succ) { std::cout << "\tbase_prover zA = ct passed\n"; }
  }

  LabradorBaseCaseProof final_proof{z_hat, T_tilde, g_tilde, h_tilde};
  std::cout << "Step 29 completed: Computed z_hat and created final proof" << std::endl;

  std::cout << "base_case_prover completed successfully!" << std::endl;
  return std::make_pair(final_proof, trs);
}

std::vector<Rq> LabradorProver::prepare_recursion_witness(
  const LabradorParam& prev_param, const LabradorBaseCaseProof& pf, uint32_t base0, size_t mu, size_t nu)
{
  // Step 1: Convert z_hat back to polynomial domain
  size_t n = prev_param.n;
  size_t r = prev_param.r;

  std::vector<Rq> z(n);
  ICICLE_CHECK(ntt(pf.z_hat.data(), pf.z_hat.size(), NTTDir::kInverse, {}, z.data()));

  // Step 2: Decompose z using base0
  std::vector<Rq> z_tilde(2 * n);
  ICICLE_CHECK(decompose(z.data(), n, base0, {}, z_tilde.data(), z_tilde.size()));

  std::vector<Rq> temp(n);
  ICICLE_CHECK(recompose(z_tilde.data(), z_tilde.size(), base0, {}, temp.data(), temp.size()));
  if (!poly_vec_eq(z.data(), temp.data(), n)) {
    throw std::runtime_error("Parameter Choice Error: z could not be recomposed from z_tilde in "
                             "prepare_recursion_witness. Consider changing base0 parameter.");
  } else {
    std::cout << "\tprepare_recursion_witness: z recomposition passes.\n";
  }
  // Step 3:
  // z0 = z_tilde[:n]
  // z1 = z_tilde[n:2*n]

  RecursionPreparer preparer{prev_param, mu, nu, base0};

  std::vector<Rq> s_prime(preparer.r_prime * preparer.n_prime, zero());

  // copy z0 = z_tilde[0 : n] →  s_prime[0 : n]
  ICICLE_CHECK(preparer.copy_like_z0(s_prime.data(), z_tilde.data()));

  // copy z1 = z_tilde[n : 2n] →  s_prime[nu * n_prime : nu * n_prime + n]
  ICICLE_CHECK(preparer.copy_like_z1(s_prime.data(), &z_tilde[n]));

  // copy t  →  s_prime[2*nu*n_prime : 2*nu*n_prime + |t|]
  ICICLE_CHECK(preparer.copy_like_t(s_prime.data(), pf.t.data()));

  // copy g  →  s_prime[(2*nu + L_t) * n_prime : ...]
  ICICLE_CHECK(preparer.copy_like_g(s_prime.data(), pf.g.data()));

  // copy h  →  s_prime[(2*nu + L_t + L_g) * n_prime : ...]
  ICICLE_CHECK(preparer.copy_like_h(s_prime.data(), pf.h.data()));

  return s_prime;
}

std::pair<std::vector<PartialTranscript>, LabradorBaseCaseProof> LabradorProver::prove()
{
  std::vector<PartialTranscript> trs;
  PartialTranscript part_trs;
  LabradorBaseCaseProof base_proof;
  LabradorInstance lab_inst_i = lab_inst;
  std::vector<Rq> S_i = S;
  for (size_t i = 0; i < NUM_REC - 1; i++) {
    std::cout << "Prover::Recursion iteration = " << i << "\n";
    LabradorBaseProver base_prover(lab_inst_i, S_i, oracle);
    std::tie(base_proof, part_trs) = base_prover.base_case_prover();
    trs.push_back(part_trs);

    // Prepare recursion problem and witness
    // NOTE: base0 needs to be large enough
    uint32_t base0 = calc_base0(lab_inst_i.param.r, OP_NORM_BOUND, lab_inst_i.param.beta);
    size_t m = lab_inst_i.param.t_len() + lab_inst_i.param.g_len() + lab_inst_i.param.h_len();
    auto [mu, nu] = compute_mu_nu(lab_inst_i.param.n, m);

    S_i = prepare_recursion_witness(lab_inst_i.param, base_proof, base0, mu, nu);
    EqualityInstance final_const = base_prover.lab_inst.equality_constraints[0];
    lab_inst_i = prepare_recursion_instance(base_prover.lab_inst.param, final_const, part_trs, base0, mu, nu);

    oracle = base_prover.oracle;

    std::cout << "\tRecursion problem prepared\n";
    std::cout << "\tn= " << lab_inst_i.param.n << ", r= " << lab_inst_i.param.r << "\n";

    if (TESTING) {
      if (lab_witness_legit(lab_inst_i, S_i)) {
        std::cout << "\tRecursion valid\n";
      } else {
        std::cout << "\tRecursion INVALID\n";
        // // Debug: check each constraint individually
        // std::cout << "\tDebugging constraints (iteration " << i << "):\n";
        // std::cout << "\tNumber of equality constraints: " << lab_inst_i.equality_constraints.size() << "\n";
        // std::cout << "\tNumber of const-zero constraints: " << lab_inst_i.const_zero_constraints.size() << "\n";
        // std::cout << "\tWitness size: " << S_i.size() << ", expected: " << lab_inst_i.param.r * lab_inst_i.param.n
        //           << "\n";

        // for (size_t j = 0; j < lab_inst_i.equality_constraints.size(); j++) {
        //   if (!witness_legit_eq(lab_inst_i.equality_constraints[j], S_i)) {
        //     std::cout << "\t\tEquality constraint " << j << " FAILED\n";
        //   } else {
        //     std::cout << "\t\tEquality constraint " << j << " passed\n";
        //   }
        // }

        // for (size_t j = 0; j < lab_inst_i.const_zero_constraints.size(); j++) {
        //   if (!witness_legit_const_zero(lab_inst_i.const_zero_constraints[j], S_i)) {
        //     std::cout << "\t\tConst-zero constraint " << j << " FAILED\n";
        //   } else {
        //     std::cout << "\t\tConst-zero constraint " << j << " passed\n";
        //   }
        // }
      }
    }
  }
  std::cout << "Prover::Recursion iteration = " << NUM_REC - 1 << "\n";
  LabradorBaseProver base_prover(lab_inst_i, S_i, oracle);
  std::tie(base_proof, part_trs) = base_prover.base_case_prover();
  trs.push_back(part_trs);

  return std::make_pair(trs, base_proof);
}
