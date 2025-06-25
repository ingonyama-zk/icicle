#include "labrador_protocol.h"
#include "icicle/lattice/labrador.h" // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h"      // For Hash

using namespace icicle::labrador;

constexpr bool TESTING = true;

/// @brief Computes the Ajtai commitment of the given input S. Views input S as matrix of vectors to be committed.
/// Vectors are arranged in the row major form. If A is the Ajtai matrix, then this outputs S@A.
/// @param ajtai_mat_seed seed for calculating entries of random Ajtai commitment matrix
/// @param seed_len length of ajtai_mat_seed
/// @param input_len length of vectors to be committed
/// @param output_len length of commitments
/// @param S data to be committed
/// @param S_len length of data to be committed. If `S_len > input_len` then S_len must be a multiple of input_len. The
/// input S will be viewed as a row major arrangement of S_len/input_len vectors to be committed.
/// @return S_len/input_len commitments of length equal to output_len arranged in row major form.
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

/// Prover uses this function to select a valid JL projection for which the norm condition is satisfied .
/// Returns (JL_i, p) such that [seed, JL_i] is the seed for the valid JL projection
/// p is the result of applying this JL projection to the witness
std::pair<size_t, std::vector<Zq>> LabradorBaseProver::select_valid_jl_proj(std::byte* seed, size_t seed_len) const
{
  size_t JL_out = lab_inst.param.JL_out;
  size_t n = lab_inst.n;
  size_t r = lab_inst.r;
  size_t d = Rq::d;

  std::vector<Zq> p(JL_out);
  size_t JL_i = 0;
  std::vector<std::byte> jl_seed(seed, seed + seed_len);
  while (true) {
    jl_seed.push_back(std::byte(JL_i));
    // create JL projection: P*(s_1, s_2, ..., s_r)
    ICICLE_CHECK(jl_projection(
      reinterpret_cast<const Zq*>(S.data()), n * r * d, jl_seed.data(), jl_seed.size(), {}, p.data(), JL_out));
    // check norm
    bool JL_check = false;
    double beta = lab_inst.param.beta;
    ICICLE_CHECK(check_norm_bound(p.data(), JL_out, eNormType::L2, uint64_t(sqrt(JL_out / 2) * beta), {}, &JL_check));

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

/// returns Q: JL_out X r X n matrix such that
/// Q(i,:,:) is the conjugation of the ith row of the JL projection viewed as a polynomial vector.
/// So that const(<Q(i,:,:), S(:,:)>) = p_i
/// JL_i needs to be the same as the one given by select_valid_jl_proj
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

// modifies the instance
// returns num_aggregation_rounds number of polynomials
std::vector<Tq> LabradorBaseProver::agg_const_zero_constraints(
  size_t num_aggregation_rounds,
  size_t JL_out,
  const std::vector<Tq>& S_hat,
  const std::vector<Tq>& G_hat,
  const std::vector<Zq>& p,
  const std::vector<Tq>& Q_hat,
  const std::vector<Zq>& psi,
  const std::vector<Zq>& omega)
{
  size_t r = lab_inst.r;
  size_t n = lab_inst.n;
  size_t d = Rq::d;
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

  std::vector<Zq> verif_test_b0(num_aggregation_rounds, Zq::zero());

  std::vector<Tq> msg3;
  for (size_t k = 0; k < num_aggregation_rounds; k++) {
    EqualityInstance new_constraint(r, n);
    std::vector<ConstZeroInstance> temp_const(lab_inst.const_zero_constraints);
    std::vector<Tq> Q_hat_copy(Q_hat);

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
    // Q_hat[j, :, :] = omega[k,j]* Q_hat[j, :, :]
    // use async_config to parallelise
    // TODO: can async with a aggregation above- leave for later
    for (size_t j = 0; j < JL_out; j++) {
      Zq omega_scalar = omega[omega_index(k, j)];

      ICICLE_CHECK(scalar_mul_vec(
        &omega_scalar, reinterpret_cast<Zq*>(&Q_hat_copy[j * n * r]), r * n * d, async_config,
        reinterpret_cast<Zq*>(&Q_hat_copy[j * n * r])));
    }
    ICICLE_CHECK(icicle_device_synchronize());
    // new_constraint.phi[i,:] += \sum_j Q_hat[j, i, :]
    for (size_t j = 0; j < JL_out; j++) {
      ICICLE_CHECK(vector_add(new_constraint.phi.data(), &Q_hat_copy[j * n * r], r * n, {}, new_constraint.phi.data()));
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

      // if (verif_test_b0[k] != b_rq.values[0]) {
      //   std::cout << "\tFail: New constraint b doesn't match verif b for idx" << k << "\n";
      // } else {
      //   std::cout << "\tPass: New constraint b matches verif b for idx" << k << "\n";
      // }
      if (!witness_legit_const_zero({r, n, new_constraint.a, new_constraint.phi, verif_test_b0[k]}, S)) {
        std::cout << "\tVerif test constraint " << k << " failed\n";
      }
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

// Modifies equality constraints
void LabradorInstance::agg_equality_constraints(const std::vector<Tq>& alpha_hat)
{ // Step 22: Say the EqualityInstances in LabradorInstance are:
  // [{a_{ij}^{(k)}; 0 ≤ i,j < r} ⊂ T_q, b^{(k)} ∈ T_q, {φ_i^{(k)} : 0 ≤ i < r} ⊂ T_q^n : 0 ≤ k < K]

  // For 0 ≤ i,j < r, the Prover computes a''_{ij}:
  const size_t K = equality_constraints.size();
  const size_t d = Tq::d;
  EqualityInstance final_const(r, n);

  VecOpsConfig async_config = default_vec_ops_config();
  async_config.is_async = true;

  // a''_{ij} = ∑_{k=0}^{K-1} α_k * a_{ij}^{(k)}

  // Compute: equality_constraints[k].a = alpha_hat[k]*equality_constraints[k].a
  for (size_t k = 0; k < K; k++) {
    // no iterations like agg_const_zero_constraints, so in place modification
    ICICLE_CHECK(matmul(
      &alpha_hat[k], 1, 1, equality_constraints[k].a.data(), 1, r * r, async_config, equality_constraints[k].a.data()));
  }
  ICICLE_CHECK(icicle_device_synchronize());
  // final_const.a = \sum_k equality_constraints[k].a
  for (size_t k = 0; k < K; k++) {
    ICICLE_CHECK(vector_add(final_const.a.data(), equality_constraints[k].a.data(), r * r, {}, final_const.a.data()));
  }

  // φ'_i = ∑_{k=0}^{K-1} α_k * φ_i^{(k)}
  // Compute: equality_constraints[k].phi = alpha_hat[k]*equality_constraints[k].phi
  for (size_t k = 0; k < K; k++) {
    ICICLE_CHECK(matmul(
      &alpha_hat[k], 1, 1, equality_constraints[k].phi.data(), 1, r * n, async_config,
      equality_constraints[k].phi.data()));
  }
  ICICLE_CHECK(icicle_device_synchronize());
  // final_const.phi = \sum_k equality_constraints[k].phi
  for (size_t k = 0; k < K; k++) {
    ICICLE_CHECK(
      vector_add(final_const.phi.data(), equality_constraints[k].phi.data(), r * n, {}, final_const.phi.data()));
  }

  // b = ∑_{k=0}^{K-1} α_k * b^{(k)}
  for (size_t k = 0; k < K; k++) {
    // Get b^{(k)} from equality constraint k (already in T_q)
    Tq b_k = equality_constraints[k].b;

    // Multiply by α_k and add to sum (T_q operations)
    Tq temp;
    ICICLE_CHECK(vector_mul(&alpha_hat[k], &b_k, 1, {}, &temp));
    ICICLE_CHECK(vector_add(&final_const.b, &temp, 1, {}, &final_const.b));
  }

  equality_constraints = {final_const};
}

// This destroys the lab_inst in LabradorBaseProver
std::pair<LabradorBaseCaseProof, PartialTranscript> LabradorBaseProver::base_case_prover()
{
  // Step 1: Pack the Witnesses into a Matrix S
  const size_t r = lab_inst.r; // Number of witness vectors
  const size_t n = lab_inst.n; // Dimension of witness vectors
  constexpr size_t d = Rq::d;

  if (TESTING) {
    std::cout << "\tTesting witness validity...";
    assert(lab_witness_legit(lab_inst, S));
    std::cout << "VALID\n";
  }

  PartialTranscript trs;
  std::cout << "Step 1 completed: Initialized variables" << std::endl;

  // Step 2: Convert S to the NTT Domain
  std::vector<Tq> S_hat(r * n);
  // Perform negacyclic NTT on the witness S
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));
  std::cout << "Step 2 completed: NTT conversion" << std::endl;

  // Step 3: S@A = T
  const std::vector<std::byte>& ajtai_seed = lab_inst.param.ajtai_seed;
  std::vector<std::byte> seed_A(ajtai_seed);
  seed_A.push_back(std::byte('0'));

  // Use ajtai_commitment to compute T_hat = S_hat @ A
  size_t kappa = lab_inst.param.kappa;
  std::vector<Tq> T_hat = ajtai_commitment(seed_A.data(), seed_A.size(), n, kappa, S_hat.data(), r * n);
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
  size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base2);
  std::vector<Rq> g_tilde(l2 * g.size());
  ICICLE_CHECK(decompose(g.data(), g.size(), base2, {}, g_tilde.data(), g_tilde.size()));
  std::cout << "Step 8 completed: Decomposed g to g_tilde" << std::endl;

  // Step 9: already done

  // Step 10: u1 = B@T_tilde + C@g_tilde
  size_t kappa1 = lab_inst.param.kappa1;
  std::vector<std::byte> seed_B(ajtai_seed), seed_C(ajtai_seed);
  seed_B.push_back(std::byte('1'));
  seed_C.push_back(std::byte('2'));

  // compute NTTs for T_tilde, g_tilde
  std::vector<Tq> T_tilde_hat(T_tilde.size()), g_tilde_hat(g_tilde.size());
  ICICLE_CHECK(ntt(T_tilde.data(), T_tilde.size(), NTTDir::kForward, {}, T_tilde_hat.data()));
  ICICLE_CHECK(ntt(g_tilde.data(), g_tilde.size(), NTTDir::kForward, {}, g_tilde_hat.data()));

  // v1 = B@T_tilde
  std::vector<Tq> v1 =
    ajtai_commitment(seed_B.data(), seed_B.size(), l1 * r * kappa, kappa1, T_tilde_hat.data(), T_tilde_hat.size());
  // v2 = C@g_tilde
  std::vector<Tq> v2 =
    ajtai_commitment(seed_C.data(), seed_C.size(), (r_choose_2)*l2, kappa1, g_tilde_hat.data(), g_tilde_hat.size());

  std::vector<Tq> u1(kappa1);
  vector_add(v1.data(), v2.data(), kappa1, {}, u1.data());
  std::cout << "Step 10 completed: Computed u1" << std::endl;

  // Step 11: hash (lab_inst, ajtai_seed, u1) to get seed1
  // hash and get a challenge
  Hash hasher = Sha3_256::create();
  // TODO: add serialization to lab_inst, ajtai_seed, u1 and put them in the placeholder
  std::vector<std::byte> seed1(hasher.output_size());
  {
    const char* hash_input = "Placeholder1";
    hasher.hash(hash_input, strlen(hash_input), {}, seed1.data());
  }
  // add u1 to the trs
  trs.u1 = u1;
  trs.seed1 = seed1;
  std::cout << "Step 11 completed: Generated seed1" << std::endl;

  // Step 12: Select a JL projection
  size_t JL_out = lab_inst.param.JL_out;
  auto [JL_i, p] = select_valid_jl_proj(seed1.data(), seed1.size());
  std::cout << "Step 12 completed: Selected JL projection" << std::endl;

  // Step 13: send (JL_i, p) to the Verifier and get a challenge
  trs.JL_i = JL_i;
  trs.p = p;

  // TODO: add serialization to p and JL_i and put them in the placeholder
  std::vector<std::byte> seed2(hasher.output_size());
  {
    const char* hash_input = "Placeholder2";
    hasher.hash(hash_input, strlen(hash_input), {}, seed2.data());
  }
  trs.seed2 = seed2;
  std::cout << "Step 13 completed: Generated seed2" << std::endl;

  // Step 14: removed
  // Step 15, 16: already done

  // Step 17: Create conjugated polynomial vectors from JL matrix rows
  std::vector<Rq> Q = compute_Q_poly(n, r, JL_out, seed1.data(), seed1.size(), JL_i);
  std::cout << "Step 17 completed: Computed Q polynomial" << std::endl;

  // Step 18: Let L be the number of constZeroInstance constraints in LabradorInstance.
  // For 0 ≤ k < ceil(128/log(q)), sample the following random vectors:
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t num_aggregation_rounds = std::ceil(128.0 / std::log2(get_q<Zq>()));

  std::vector<Zq> psi(num_aggregation_rounds * L), omega(num_aggregation_rounds * JL_out);

  // sample psi
  std::vector<std::byte> psi_seed(seed2);
  psi_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(psi_seed.data(), psi_seed.size(), false, {}, psi.data(), psi.size()));

  // Sample omega
  std::vector<std::byte> omega_seed(seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(omega_seed.data(), omega_seed.size(), false, {}, omega.data(), omega.size()));

  trs.psi = psi;
  trs.omega = omega;
  std::cout << "Step 18 completed: Sampled psi and omega" << std::endl;

  // Step 19: Aggregate ConstZeroInstance constraints
  std::vector<Tq> Q_hat(JL_out * r * n);
  // Q_hat = NTT(Q)
  ICICLE_CHECK(ntt(Q.data(), Q.size(), NTTDir::kForward, {}, Q_hat.data()));

  if (TESTING) {
    bool Q_testing = true;
    for (size_t l = 0; l < JL_out; l++) {
      std::vector<Tq> a{r * r, zero()}, phi{&Q_hat[l * r * n], &Q_hat[(l + 1) * r * n]};

      ConstZeroInstance cz{r, n, a, phi, Zq::neg(p[l])};
      if (!witness_legit_const_zero(cz, S)) {
        std::cout << "\tQ-constraint-check fails for " << l << "\n";
        Q_testing = false;
        break;
      };
    }
    if (Q_testing) { std::cout << "\tQ-constraint-check passed... " << "\n"; }
  }

  std::vector<Tq> msg3 = agg_const_zero_constraints(num_aggregation_rounds, JL_out, S_hat, G_hat, p, Q_hat, psi, omega);
  std::cout << "Step 19 completed: Aggregated ConstZeroInstance constraints" << std::endl;

  if (TESTING) {
    std::cout << "\tTesting witness validity...";
    assert(lab_witness_legit(lab_inst, S));
    std::cout << "VALID\n";
  }

  // Step 20: seed3 = hash(seed2, msg3)
  // TODO: add serialization to msg3 and put them in the placeholder
  std::vector<std::byte> seed3(hasher.output_size());
  hasher.hash("Placeholder3", 12, {}, seed3.data());

  trs.b_agg = msg3;
  trs.seed3 = seed3;
  std::cout << "Step 20 completed: Generated seed3" << std::endl;

  // Step 21: Sample random polynomial vectors α using seed3
  // Let K be the number of EqualityInstances in the LabradorInstance
  const size_t K = lab_inst.equality_constraints.size();

  std::vector<Tq> alpha_hat(K);
  std::vector<std::byte> alpha_seed(seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(alpha_seed.data(), alpha_seed.size(), false, {}, alpha_hat.data(), K));

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
  std::vector<std::byte> seed_D(ajtai_seed);
  seed_D.push_back(std::byte('3'));
  // u2 = D@h_tilde
  std::vector<Tq> u2 =
    ajtai_commitment(seed_D.data(), seed_D.size(), l3 * r_choose_2, kappa2, h_tilde_hat.data(), h_tilde_hat.size());
  std::cout << "Step 26 completed: Computed u2 commitment" << std::endl;

  // Step 27:
  // add u2 to the trs
  trs.u2 = u2;

  // TODO: add serialization to u2 and put them in the placeholder
  std::vector<std::byte> seed4(hasher.output_size());
  hasher.hash("Placeholder4", 12, {}, seed4.data());

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
    std::vector<Tq> zA_hat = ajtai_commitment(seed_A.data(), seed_A.size(), n, kappa, z_hat.data(), n);

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

  LabradorBaseCaseProof final_proof{lab_inst.equality_constraints[0], z_hat, T_tilde, g_tilde, h_tilde};
  std::cout << "Step 29 completed: Computed z_hat and created final proof" << std::endl;

  std::cout << "base_case_prover completed successfully!" << std::endl;
  return std::make_pair(final_proof, trs);
}

std::vector<Rq> LabradorProver::prepare_recursion_witness(
  const PartialTranscript& trs, const LabradorBaseCaseProof& pf, size_t base0, size_t mu, size_t nu)
{
  // Step 1: Convert z_hat back to polynomial domain
  size_t n = pf.final_const.n;
  size_t r = pf.final_const.r;

  std::vector<Rq> z(n);
  ICICLE_CHECK(ntt(pf.z_hat.data(), pf.z_hat.size(), NTTDir::kInverse, {}, z.data()));

  // Step 2: Decompose z using base0
  size_t l0 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base0);

  // Decompose all elements first
  std::vector<Rq> z_tilde(l0 * n);
  ICICLE_CHECK(decompose(z.data(), n, base0, {}, z_tilde.data(), z_tilde.size()));
  // Keep only first 2n elements- all the rest should be 0
  z_tilde.resize(2 * n);

  // Step 3:
  // z0 = z_tilde[:n]
  // z1 = z_tilde[n:2*n]

  size_t m = pf.t.size() + pf.g.size() + pf.h.size();

  // Step 4, 5:
  size_t n_prime = std::max(std::ceil((double)n / nu), std::ceil((double)m / mu));

  // Step 6
  // we will view s_prime as a multidimensional array. At the base level it consists of n_prime length vectors
  std::vector<Rq> s_prime;

  for (size_t i = 0; i < n; i++) {
    s_prime.push_back(z_tilde[i]);
  }
  for (size_t i = n; i < nu * n_prime; i++) {
    s_prime.push_back(Rq());
  }
  // now s_prime is nu*n_prime long and can be viewed as a nu long array of n_prime dimension Tq vectors

  for (size_t i = 0; i < n; i++) {
    s_prime.push_back(z_tilde[n + i]);
  }
  for (size_t i = n; i < nu * n_prime; i++) {
    s_prime.push_back(Rq());
  }
  // now s_prime is 2*nu*n_prime long and can be viewed as a 2*nu long array of n_prime dimension Tq vectors

  // add the polynomials in t to s_prime and zero pad to make them L_t*n_prime length
  size_t L_t = (pf.t.size() + n_prime - 1) / n_prime;
  for (size_t i = 0; i < pf.t.size(); i++) {
    s_prime.push_back(pf.t[i]);
  }
  for (size_t i = pf.t.size(); i < L_t * n_prime; i++) {
    s_prime.push_back(Rq());
  }

  // add the polynomials in g to s_prime and zero pad to make them L_g*n_prime length
  size_t L_g = (pf.g.size() + n_prime - 1) / n_prime;
  for (size_t i = 0; i < pf.g.size(); i++) {
    s_prime.push_back(pf.g[i]);
  }
  for (size_t i = pf.g.size(); i < L_g * n_prime; i++) {
    s_prime.push_back(Rq());
  }

  // add the polynomials in h to s_prime and zero pad to make them L_h*n_prime length
  size_t L_h = (pf.h.size() + n_prime - 1) / n_prime;
  for (size_t i = 0; i < pf.h.size(); i++) {
    s_prime.push_back(pf.h[i]);
  }
  for (size_t i = pf.h.size(); i < L_h * n_prime; i++) {
    s_prime.push_back(Rq());
  }
  return s_prime;
}

/// Returns the LabradorInstance for recursion problem
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

  size_t t_len = prev_param.t_len(r);
  size_t g_len = prev_param.g_len(r);
  size_t h_len = prev_param.h_len(r);

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
    new_ajtai_seed,
    prev_param.kappa,      // kappa
    prev_param.kappa1,     // kappa1
    prev_param.kappa2,     // kappa2,
    prev_param.base1,      // base1
    prev_param.base2,      // base2
    prev_param.base3,      // base3
    100 * prev_param.beta, // beta
  };
  LabradorInstance recursion_instance{r_prime, n_prime, recursion_param};

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

std::pair<std::vector<PartialTranscript>, LabradorBaseCaseProof> LabradorProver::prove()
{
  std::vector<PartialTranscript> trs;
  PartialTranscript part_trs;
  LabradorBaseCaseProof base_proof(lab_inst.r, lab_inst.n);
  for (size_t i = 0; i < NUM_REC; i++) {
    LabradorBaseProver base_prover(lab_inst, S);
    std::tie(base_proof, part_trs) = base_prover.base_case_prover();
    // TODO: figure out param using Lattirust code
    size_t base0 = 1 << 8, mu = 1 << 8, nu = 1 << 8;
    S = prepare_recursion_witness(part_trs, base_proof, base0, mu, nu);
    lab_inst = prepare_recursion_instance(base_prover.lab_inst.param, base_proof.final_const, part_trs, base0, mu, nu);
    trs.push_back(part_trs);
  }
  return std::make_pair(trs, base_proof);
}

// TODO: maybe make BaseProof more consistent by making everything Rq, since we have to convert z_hat to Rq before norm
// check anyway
bool LabradorBaseVerifier::_verify_base_proof() const
{
  size_t n = lab_inst.n;
  size_t r = lab_inst.r;
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
  if (!(poly_vec_eq(u1.data(), trs.u1.data(), kappa1))) {
    std::cout << "t_tilde, g_tilde don't open u1 \n";
    return false;
  }

  size_t kappa2 = lab_inst.param.kappa2;
  // u2 = D@h_tilde
  std::vector<Tq> u2 =
    ajtai_commitment(seed_D.data(), seed_D.size(), h_tilde_hat.size(), kappa2, h_tilde_hat.data(), h_tilde_hat.size());

  // check h_tilde opens to u2 in trs
  if (!(poly_vec_eq(u2.data(), trs.u2.data(), kappa2))) {
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

// === Main driver ===

int main(int argc, char* argv[])
{
  ICICLE_LOG_INFO << "Labrador example";
  try_load_and_set_backend_device(argc, argv);

  const int64_t q = get_q<Zq>();

  // TODO use icicle_malloc() instead of std::vector. Consider a DeviceVector<T> that behaves like std::vector

  // randomize the witness Si with low norm
  const size_t n = 1 << 5;
  const size_t r = 1 << 3;
  constexpr size_t d = Rq::d;
  std::vector<Rq> S = rand_poly_vec(r * n, 1);
  EqualityInstance eq_inst = create_rand_eq_inst(n, r, S);
  assert(witness_legit_eq(eq_inst, S));
  ConstZeroInstance const_zero_inst = create_rand_const_zero_inst(n, r, S);
  assert(witness_legit_const_zero(const_zero_inst, S));

  // Use current time (milliseconds since epoch) as a unique Ajtai seed
  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::string ajtai_seed_str = std::to_string(millis);
  std::cout << "Ajtai seed = " << ajtai_seed_str << std::endl;
  LabradorParam param{
    {reinterpret_cast<const std::byte*>(ajtai_seed_str.data()),
     reinterpret_cast<const std::byte*>(ajtai_seed_str.data()) + ajtai_seed_str.size()},
    1 << 4,    // kappa
    1 << 4,    // kappa1
    1 << 4,    // kappa2,
    1 << 16,   // base1
    1 << 16,   // base2
    1 << 16,   // base3
    n * r * d, // beta
  };
  LabradorInstance lab_inst{r, n, param};
  lab_inst.add_equality_constraint(eq_inst);
  lab_inst.add_const_zero_constraint(const_zero_inst);

  LabradorBaseProver base_prover{lab_inst, S};
  auto [base_proof, trs] = base_prover.base_case_prover();

  LabradorInstance verif_lab_inst{r, n, param};
  verif_lab_inst.add_equality_constraint(eq_inst);
  verif_lab_inst.add_const_zero_constraint(const_zero_inst);

  LabradorBaseVerifier base_verifier{verif_lab_inst, trs, base_proof};
  bool verification_result = base_verifier._verify_base_proof();

  if (verification_result) {
    std::cout << "Base proof verification passed\n";
  } else {
    std::cout << "Base proof verification failed\n";
  }

  std::cout << "Hello\n";
  return 0;
}

// n*r = 2^30
// r^2 = n
// r= 2^10, n=2^20

// A*B
// A: k X n
// B: n X r

// Zq, base = q^{1/t} t=2,4,6
