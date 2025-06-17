#include "labrador_protocol.h"
#include "icicle/lattice/labrador.h" // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h"      // For Hash

using namespace icicle::labrador;

/// @brief Computes the Ajtai input of the given input S. Views input S as matrix of vectors to be committed. Vectors
/// are arranged in the row major form.
/// @param ajtai_mat_seed seed for calculating entries of random Ajtai commitment matrix
/// @param seed_len length of ajtai_mat_seed
/// @param input_len length of vectors to be committed
/// @param output_len length of commitments
/// @param S data to be committed
/// @param S_len length of data to be committed. If `S_len > input_len` then S_len must be a multiple of input_len. The
/// input S will be viewed as a row major arrangement of S_len/input_len vectors to be committed.
/// @return S_len/input_len commitments of length equal to output_len arranged in row major form.
Tq* ajtai_commitment(
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
  return comm.data();
}

std::vector<Tq> LabradorInstance::aggregate_const_zero_inst(
  size_t num_aggregation_rounds,
  size_t JL_out,
  const std::vector<Tq>& S_hat,
  const std::vector<Tq>& g_hat,
  const std::vector<Rq>& Q,
  const std::vector<Zq>& psi,
  const std::vector<Zq>& omega)
{
  size_t d = Rq::d;
  const size_t L = const_zero_constraints.size();

  // indexes into a multidim array of dim = r X JL_out X n
  auto Q_index = [n = this->n, JL_out](size_t i, size_t j, size_t k) { return (i * JL_out * n + j * n + k); };
  // indexes into multidim arrays: psi[k][l] and omega[k][l]
  auto psi_index = [L](size_t k, size_t l) { return k * L + l; };
  auto omega_index = [JL_out](size_t k, size_t l) { return k * JL_out + l; };

  std::vector<Tq> msg3;
  for (size_t k = 0; k < num_aggregation_rounds; k++) {
    EqualityInstance new_constraint(r, n);

    // Compute a''_{ij} = sum_{l=0}^{L-1} psi^{(k)}(l) * a'_{ij}^{(l)}
    for (size_t i = 0; i < r; i++) {
      for (size_t j = 0; j < r; j++) {
        Tq sum = Tq(); // Initialize to zero polynomial

        // TODO vectorize loop
        for (size_t l = 0; l < L; l++) {
          // Get psi^{(k)}(l) as scalar
          Zq psi_scalar = psi[psi_index(k, l)];

          // Get a_{ij}^{(l)} from const_zero_constraints
          Tq a_ij_l = const_zero_constraints[l].a[i][j];

          // Scalar multiply and add: sum += psi_scalar * a_ij_l
          // TODO: use vector_mul<Rq,Zq> and vector_sum<Rq> to aggregate in a vectorized way
          Tq temp;
          ICICLE_CHECK(scalar_mul_vec(&psi_scalar, a_ij_l.values, d, {}, temp.values));
          ICICLE_CHECK(vector_add(sum.values, temp.values, d, {}, sum.values));
        }

        new_constraint.a[i][j] = sum;
      }
    }

    // Compute varphi'_i^{(k)} = sum_{l=0}^{L-1} psi^{(k)}(l) * phi'_i^{(l)} + sum_{l=0}^{255} omega^{(k)}(l) * q_{il}
    for (size_t i = 0; i < r; i++) {
      // First sum: over const_zero_constraints
      for (size_t l = 0; l < L; l++) {
        Zq psi_scalar = psi[psi_index(k, l)];

        for (size_t m = 0; m < n; m++) {
          Tq phi_il_m = const_zero_constraints[l].phi[i][m];

          // phi_prime[i,m] += psi_scalar * phi_il_m
          Tq temp;
          ICICLE_CHECK(scalar_mul_vec(&psi_scalar, phi_il_m.values, d, {}, temp.values));
          ICICLE_CHECK(
            vector_add(new_constraint.phi[i][m].values, temp.values, d, {}, new_constraint.phi[i][m].values));
        }
      }

      // Second sum: over JL projection rows (256 rows)
      for (size_t l = 0; l < JL_out; l++) { // JL_out = 256
        Zq omega_scalar = omega[omega_index(k, l)];

        // TODO: can vectorize the loop?
        for (size_t m = 0; m < n; m++) {
          // q_{il} is stored in Q[i][l][:]
          Rq q_ilm = Q[Q_index(i, l, m)];
          Tq q_ilm_hat;
          ICICLE_CHECK(ntt(&q_ilm, 1, NTTDir::kForward, {}, &q_ilm_hat));

          // phi_i_k[m] += omega_scalar * q_ilm
          Tq temp;
          ICICLE_CHECK(scalar_mul_vec(&omega_scalar, q_ilm_hat.values, d, {}, temp.values));
          ICICLE_CHECK(
            vector_add(new_constraint.phi[i][m].values, temp.values, d, {}, new_constraint.phi[i][m].values));
        }
      }
    }

    // Compute B^{(k)} = (sum_{i<j} (a''_{ij}^{(k)} + a''_{ji}^{(k)}) * g_{ij} + sum_i a''_{ii}^{(k)} * g_{ii})
    //                       + sum_i <phi'_i^{(k)}, s_i>

    // First part: sum over g terms

    std::vector<Tq> a_vec;
    // shape a_vec like g
    for (size_t i = 0; i < r; i++) {
      for (size_t j = i; j < r; j++) {
        if (i == j) {
          a_vec.push_back(new_constraint.a[i][i]);
        } else {
          // Off-diagonal: (a''_{ij} + a''_{ji})
          Tq a_ij = new_constraint.a[i][j];
          Tq a_ji = new_constraint.a[j][i];
          Tq temp;
          ICICLE_CHECK(vector_add(a_ij.values, a_ji.values, d, {}, temp.values));
          a_vec.push_back(temp);
        }
      }
    }

    ICICLE_CHECK(matmul(g_hat.data(), 1, g_hat.size(), a_vec.data(), a_vec.size(), 1, {}, &new_constraint.b));

    // Second part: sum_i <phi'_i^{(k)}, s_i>
    for (size_t i = 0; i < r; i++) {
      // Compute inner product <phi'_i^{(k)}, s_i>
      Tq prod;
      ICICLE_CHECK(matmul(new_constraint.phi[i].data(), 1, n, &S_hat[i * n], n, 1, {}, &prod));
      ICICLE_CHECK(vector_add(new_constraint.b.values, prod.values, d, {}, new_constraint.b.values));
    }

    // Add the EqualityInstance to LabradorInstance
    add_equality_constraint(new_constraint);

    // Send B^(k) to the Verifier
    msg3.push_back(new_constraint.b);
  }
  // delete the const zero constraints
  const_zero_constraints.clear();
  const_zero_constraints.shrink_to_fit();

  return msg3;
}

LabradorRecursionRawInstance LabradorProtocol::base_prover(
  LabradorInstance& lab_inst,
  const std::vector<std::byte>& ajtai_seed,
  const std::vector<Rq>& S,
  std::vector<Zq>& proof)
{
  // Step 1: Pack the Witnesses into a Matrix S
  const size_t r = lab_inst.r; // Number of witness vectors
  const size_t n = lab_inst.n; // Dimension of witness vectors
  constexpr size_t d = Rq::d;
  // Ensure S is of the correct size
  if (S.size() != r * n) { throw std::invalid_argument("S must have size r * n"); }

  // Step 2: Convert S to the NTT Domain
  std::vector<Tq> S_hat(r * n);
  // Perform negacyclic NTT on the witness S
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

  // Step 3: S@A = T
  std::vector<std::byte> seed_A(ajtai_seed);
  seed_A.push_back(std::byte('0'));

  // Use ajtai_commitment to compute T_hat = S_hat @ A
  Tq* T_hat_ptr = ajtai_commitment(seed_A.data(), seed_A.size(), n, kappa, S_hat.data(), r * n);
  std::vector<Tq> T_hat(T_hat_ptr, T_hat_ptr + r * kappa);

  // Step 4: already done

  // Step 5: Convert T_hat to Rq
  std::vector<Rq> T(r * kappa);
  // Perform negacyclic INTT
  ICICLE_CHECK(ntt(T_hat.data(), r * kappa, NTTDir::kInverse, {}, T.data()));

  // Step 6: decompose T to T_tilde
  size_t l1 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base1);
  std::vector<Rq> T_tilde(l1 * r * kappa);
  ICICLE_CHECK(decompose(T.data(), r * kappa, base1, {}, T_tilde.data(), T_tilde.size()));

  // Step 7: compute g
  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  size_t r_choose_2 = (r * (r + 1)) / 2;
  std::vector<Rq> g(r_choose_2);
  std::vector<Tq> g_hat;
  for (size_t i = 0; i < r; i++) {
    for (size_t j = i; j < r; j++) {
      g_hat.push_back(G_hat[i * r + j]);
    }
  }

  ICICLE_CHECK(ntt(g_hat.data(), r_choose_2, NTTDir::kInverse, {}, g.data()));

  // Step 8: decompose g to g_tilde
  size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base2);
  std::vector<Rq> g_tilde(l2 * g.size());
  ICICLE_CHECK(decompose(g.data(), g.size(), base2, {}, g_tilde.data(), g_tilde.size()));

  // Step 9: already done

  // Step 10: u1 = B@T_tilde + C@g_tilde
  // Generate B, C
  // TODO: change this so that B,C need not be computed and stored
  std::vector<Tq> B(kappa1 * l1 * r * kappa), C(kappa1 * ((r * (r + 1)) / 2) * l2);

  std::vector<std::byte> seed_B(ajtai_seed), seed_C(ajtai_seed);
  seed_B.push_back(std::byte('1'));
  seed_C.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(seed_B.data(), seed_B.size(), false, {}, B.data(), B.size()));
  ICICLE_CHECK(random_sampling(seed_C.data(), seed_C.size(), false, {}, C.data(), C.size()));

  // compute NTTs for T_tilde, g_tilde
  std::vector<Tq> T_tilde_hat(T_tilde.size()), g_tilde_hat(g_tilde.size());
  ICICLE_CHECK(ntt(T_tilde.data(), T_tilde.size(), NTTDir::kForward, {}, T_tilde_hat.data()));
  ICICLE_CHECK(ntt(g_tilde.data(), g_tilde.size(), NTTDir::kForward, {}, g_tilde_hat.data()));

  std::vector<Tq> u1(kappa1), v1(kappa1), v2(kappa1);
  // v1 = B@T_tilde
  ICICLE_CHECK(matmul(B.data(), kappa1, l1 * r * kappa, T_tilde_hat.data(), T_tilde_hat.size(), 1, {}, v1.data()));
  // v2 = C@g_tilde
  ICICLE_CHECK(
    matmul(C.data(), kappa1, ((r * (r + 1)) / 2) * l2, g_tilde_hat.data(), g_tilde_hat.size(), 1, {}, v2.data()));
  vector_add(v1.data(), v2.data(), kappa1, {}, u1.data());

  // Step 11: hash (lab_inst, ajtai_seed, u1) to get seed1
  // add u1 to the proof
  // TODO: this loop should be flattened and use icicle_copy() to handle device memory
  for (size_t i = 0; i < kappa1; i++) {
    proof.insert(proof.end(), u1[i].values, u1[i].values + d);
  }
  // hash and get a challenge
  Hash hasher = Sha3_256::create();
  // TODO: add serialization to lab_inst, ajtai_seed, u1 and put them in the placeholder
  std::vector<std::byte> seed1(hasher.output_size());
  {
    const char* hash_input = "Placeholder1";
    hasher.hash(hash_input, strlen(hash_input), {}, seed1.data());
  }

  // Step 12: Select a JL projection
  std::vector<Zq> p(JL_out, Zq::from(0));
  size_t JL_i = 0;
  // TODO:convert this to just 1 call of JL projection
  while (true) {
    std::vector<std::byte> base_jl_seed(seed1);
    base_jl_seed.push_back(std::byte(JL_i));

    for (size_t j = 0; j < r; j++) {
      // add byte j to the seed
      std::vector<std::byte> jl_seed(base_jl_seed);
      jl_seed.push_back(std::byte(j));

      std::vector<Zq> s_projection(JL_out);
      // create JL projection: P_j*s_j
      ICICLE_CHECK(jl_projection(
        reinterpret_cast<const Zq*>(S.data() + j * n), n * d, jl_seed.data(), jl_seed.size(), {}, s_projection.data(),
        s_projection.size()));
      // add output to p
      vector_add(p.data(), s_projection.data(), s_projection.size(), {}, p.data());
    }
    // check norm
    bool JL_check = false;
    ICICLE_CHECK(check_norm_bound(p.data(), JL_out, eNormType::L2, uint64_t(sqrt(JL_out / 2) * beta), {}, &JL_check));

    if (JL_check) {
      break;
    } else {
      p.assign(p.size(), Zq::from(0));
      JL_i++;
    }
  }
  // at the end JL projection is defined by JL_i and p is the projection output

  // Step 13: send (JL_i, p) to the Verifier and get a challenge
  proof.push_back(Zq::from(JL_i));
  proof.insert(proof.end(), p.begin(), p.end());

  // TODO: add serialization to p and JL_i and put them in the placeholder
  std::vector<std::byte> seed2(hasher.output_size());
  {
    const char* hash_input = "Placeholder2";
    hasher.hash(hash_input, strlen(hash_input), {}, seed2.data());
  }
  // Step 14: removed
  // Step 15, 16: already done

  // Step 17: Create conjugated polynomial vectors from JL matrix rows
  std::vector<Rq> Q(r * JL_out * n);
  // indexes into a multidim array of dim = r X JL_out X n
  auto Q_index = [n, this](size_t i, size_t j, size_t k) { return (i * JL_out * n + j * n + k); };
  for (size_t i = 0; i < r; i++) {
    // Create seed for P_i matrix (same as in step 12)
    std::vector<std::byte> base_jl_seed(seed1);
    base_jl_seed.push_back(std::byte(JL_i));
    base_jl_seed.push_back(std::byte(i));

    // compute the PI matrix, conjugated in Rq
    ICICLE_CHECK(get_jl_matrix_rows<Rq>(
      base_jl_seed.data(), base_jl_seed.size(),
      n * d,  // row_size (M = n*d)
      0,      // row_index
      JL_out, // num_rows
      true,   // conjugate
      {},     // config
      Q.data() + i * JL_out * n));
  }

  // Step 18: Let L be the number of constZeroInstance constraints in LabradorInstance.
  // For 0 ≤ k < ceil(128/log(q)), sample the following random vectors:
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t num_aggregation_rounds = std::ceil(128.0 / std::log2(get_q<Zq>()));

  std::vector<Zq> psi(num_aggregation_rounds * L), omega(num_aggregation_rounds * JL_out);
  // indexes into multidim arrays: psi[k][l] and omega[k][l]
  auto psi_index = [L](size_t k, size_t l) { return k * L + l; };
  auto omega_index = [this](size_t k, size_t l) { return k * JL_out + l; };

  // sample psi_k
  std::vector<std::byte> psi_seed(seed2);
  psi_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(psi_seed.data(), psi_seed.size(), false, {}, psi.data(), psi.size()));

  // Sample omega_k
  std::vector<std::byte> omega_seed(seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(omega_seed.data(), omega_seed.size(), false, {}, omega.data(), omega.size()));

  // Step 19: Aggregate ConstZeroInstance constraints
  // For every 0 ≤ k < ceil(128/log(q)) compute aggregated constraints

  std::vector<Tq> msg3 =
    lab_inst.aggregate_const_zero_inst(num_aggregation_rounds, JL_out, S_hat, g_hat, Q, psi, omega);

  // Step 20: seed3 = hash(seed2, msg3)
  // TODO: add serialization to msg3 and put them in the placeholder
  std::vector<std::byte> seed3(hasher.output_size());
  hasher.hash("Placeholder3", 12, {}, seed3.data());

  proof.insert(
    proof.end(), reinterpret_cast<const Zq*>(msg3.data()),
    reinterpret_cast<const Zq*>(msg3.data() + num_aggregation_rounds));

  // Step 21: Sample random polynomial vectors α using seed3
  // Let K be the number of EqualityInstances in the LabradorInstance
  const size_t K = lab_inst.equality_constraints.size();

  std::vector<Tq> alpha_hat(K);
  std::vector<std::byte> alpha_seed(seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(alpha_seed.data(), alpha_seed.size(), false, {}, alpha_hat.data(), K));

  // Step 22: Say the EqualityInstances in LabradorInstance are:
  // [{a_{ij}^{(k)}; 0 ≤ i,j < r} ⊂ T_q, b^{(k)} ∈ T_q, {φ_i^{(k)} : 0 ≤ i < r} ⊂ T_q^n : 0 ≤ k < K]

  // For 0 ≤ i,j < r, the Prover computes a''_{ij}:
  std::vector<std::vector<Tq>> a_final(r, std::vector<Tq>(r, Tq()));
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < r; j++) {
      // a''_{ij} = ∑_{k=0}^{K-1} α_k * a_{ij}^{(k)} (multiplication in T_q)
      for (size_t k = 0; k < K; k++) {
        // Get a_{ij}^{(k)} from equality constraint k (already in T_q)
        Tq a_ij_k = lab_inst.equality_constraints[k].a[i][j];

        // Multiply by α_k and add to sum (T_q operations)
        Tq temp;
        ICICLE_CHECK(vector_mul(alpha_hat[k].values, a_ij_k.values, d, {}, temp.values));
        ICICLE_CHECK(vector_add(a_final[i][j].values, temp.values, d, {}, a_final[i][j].values));
      }
    }
  }

  // For 0 ≤ i < r, the Prover computes φ'_i:
  std::vector<std::vector<Tq>> phi_final(r, std::vector<Tq>(n, Tq()));
  for (size_t i = 0; i < r; i++) {
    for (size_t m = 0; m < n; m++) {
      // φ'_i[m] = ∑_{k=0}^{K-1} α_k * φ_i^{(k)}[m] (multiplication in T_q)
      for (size_t k = 0; k < K; k++) {
        // Get φ_i^{(k)}[m] from equality constraint k (already in T_q)
        Tq phi_i_k_m = lab_inst.equality_constraints[k].phi[i][m];

        // Multiply by α_k and add to sum (T_q operations)
        Tq temp;
        ICICLE_CHECK(vector_mul(alpha_hat[k].values, phi_i_k_m.values, d, {}, temp.values));
        ICICLE_CHECK(vector_add(phi_final[i][m].values, temp.values, d, {}, phi_final[i][m].values));
      }
    }
  }

  // The Prover also computes b':
  Tq b_final;

  for (size_t k = 0; k < K; k++) {
    // Get b^{(k)} from equality constraint k (already in T_q)
    Tq b_k = lab_inst.equality_constraints[k].b;

    // Multiply by α_k and add to sum (T_q operations)
    Tq temp;
    ICICLE_CHECK(vector_mul(alpha_hat[k].values, b_k.values, d, {}, temp.values));
    ICICLE_CHECK(vector_add(b_final.values, temp.values, d, {}, b_final.values));
  }

  // roll a_final, phi_final, b_final into a single EqualityInstance
  EqualityInstance final_const(r, n, a_final, phi_final, b_final);

  // Step 23: For 0 ≤ i ≤ j < r, the Prover computes:
  // h_{ij} = 2^{-1}(<φ'_i, s_j> + <φ'_j, s_i>) ∈ R_q
  // Alternatively, view as a matrix multiplication between matrix
  // Λ = (φ'_0|φ'_1|···|φ'_{r-1})^T ∈ R_q^{r×n} and S ∈ R_q^{r×n} defined earlier.
  // Let H ∈ R_q^{r×r}, such that H_{ij} = h_{ij}
  // Then, H = 2^{-1}(Λ @ S^T + (Λ @ S^T)^T)

  // Construct Lambda_hat
  std::vector<Tq> Lambda_hat(r * n);
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < n; j++) {
      Lambda_hat[i * n + j] = phi_final[i][j];
    }
  }

  // Compute Λ @ S^T using the transposed S_hat
  std::vector<Tq> LS_hat(r * r);
  ICICLE_CHECK(matmul(Lambda_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, LS_hat.data()));

  // Convert back to Rq domain
  std::vector<Rq> LS(r * r);
  ICICLE_CHECK(ntt(LS_hat.data(), r * r, NTTDir::kInverse, {}, LS.data()));

  // Compute H = 2^{-1}(LS + LS^T)
  std::vector<Rq> H;
  Zq two_inv = Zq::inverse(Zq::from(2)); // 2^{-1} in Z_q

  for (size_t i = 0; i < r; i++) {
    // only upper triangular elements
    for (size_t j = i; j < r; j++) {
      // H[i][j] = 2^{-1} * (LS[i][j] + LS[j][i])
      Rq temp;
      ICICLE_CHECK(vector_add(LS[i * r + j].values, LS[j * r + i].values, d, {}, temp.values));
      ICICLE_CHECK(scalar_mul_vec(&two_inv, temp.values, d, {}, temp.values));
      H.push_back(temp);
    }
  }

  // Step 24: Decompose h
  size_t l3 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base3));

  std::vector<Rq> H_tilde(l3 * H.size());
  ICICLE_CHECK(decompose(H.data(), H.size(), base3, {}, H_tilde.data(), H_tilde.size()));
  std::vector<Tq> H_tilde_hat(H_tilde.size());
  ICICLE_CHECK(ntt(H_tilde.data(), H_tilde.size(), NTTDir::kForward, {}, H_tilde_hat.data()));

  // Step 25: already done
  // Step 26: commit to H_tilde
  // TODO: change this so that D need not be computed and stored
  std::vector<Tq> D(kappa2 * l3 * ((r * (r + 1)) / 2));

  std::vector<std::byte> seed_D(ajtai_seed);
  seed_D.push_back(std::byte('3'));
  ICICLE_CHECK(random_sampling(seed_D.data(), seed_D.size(), false, {}, D.data(), D.size()));

  std::vector<Tq> u2(kappa2);
  // u2 = D@H_tilde
  ICICLE_CHECK(
    matmul(D.data(), kappa2, l3 * ((r * (r + 1)) / 2), H_tilde_hat.data(), H_tilde_hat.size(), 1, {}, u2.data()));

  // Step 27:
  // add u2 to the proof
  for (size_t i = 0; i < kappa2; i++) {
    proof.insert(proof.end(), u2[i].values, u2[i].values + d);
  }
  // TODO: add serialization to u2 and put them in the placeholder
  std::vector<std::byte> seed4(hasher.output_size());
  hasher.hash("Placeholder4", 12, {}, seed4.data());

  // Step 28: sampling low operator norm challenges
  std::vector<Rq> challenge(r);
  std::vector<size_t> j_ch(r, 0);
  for (size_t i = 0; i < r; i++) {
    while (true) {
      std::vector<std::byte> ch_seed(seed4);
      ch_seed.push_back(std::byte(i));
      ch_seed.push_back(std::byte(j_ch[i]));
      ICICLE_CHECK(sample_challenge_polynomials(ch_seed.data(), ch_seed.size(), {1, 2}, {31, 10}, challenge[i]));

      bool norm_bound = false;
      ICICLE_CHECK(check_norm_bound(challenge[i].values, d, eNormType::Lop, 15, {}, &norm_bound));

      if (norm_bound) {
        break;
      } else {
        j_ch[i]++;
      }
    }
  }

  std::vector<Tq> challenge_hat(r);
  ICICLE_CHECK(ntt(challenge.data(), challenge.size(), NTTDir::kForward, {}, challenge_hat.data()));

  // Step 29: Compute z_hat
  std::vector<Tq> z_hat(n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < r; j++) {
      Tq temp;
      ICICLE_CHECK(vector_mul(challenge_hat[j].values, S_hat[j * n + i].values, d, {}, temp.values));
      ICICLE_CHECK(vector_add(z_hat[i].values, temp.values, d, {}, z_hat[i].values));
    }
  }
  LabradorRecursionRawInstance raw_inst{final_const, u1, u2, challenge_hat, z_hat, T_tilde, g_tilde, H_tilde};

  return raw_inst;
}

/// Returns the LabradorInstance for recursion problem and the witness for it
///
/// ajtai_seed: is the same seed used for Ajtai hashing in `base_prover`
///
/// rec_inst: contains the recursion problem and the witness for it
///
/// mu, nu: parameters for recursion
std::pair<LabradorInstance, std::vector<Rq>> LabradorProtocol::prepare_recursive_problem(
  std::vector<std::byte> ajtai_seed, LabradorRecursionRawInstance rec_inst, size_t mu, size_t nu)
{
  EqualityInstance final_const = rec_inst.final_const;
  std::vector<Tq> u1 = rec_inst.u1;
  std::vector<Tq> u2 = rec_inst.u2;
  std::vector<Tq> challenges_hat = rec_inst.challenges_hat;
  std::vector<Tq> z_hat = rec_inst.z_hat;
  std::vector<Rq> t = rec_inst.t;
  std::vector<Rq> g = rec_inst.g;
  std::vector<Rq> h = rec_inst.h;
  const size_t r = final_const.r;
  const size_t n = final_const.n;
  constexpr size_t d = Rq::d;

  // Step 1: Convert z_hat back to polynomial domain
  std::vector<Rq> z(n);
  ICICLE_CHECK(ntt(z_hat.data(), z_hat.size(), NTTDir::kInverse, {}, z.data()));

  // Step 2: Decompose z using base0
  size_t l0 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base0));

  // Decompose all elements first
  std::vector<Rq> z_tilde(l0 * n);
  ICICLE_CHECK(decompose(z.data(), n, base0, {}, z_tilde.data(), z_tilde.size()));
  // Keep only first 2n elements- all the rest should be 0
  z_tilde.resize(2 * n);

  // Step 3:
  // z0 = z_tilde[:n]
  // z1 = z_tilde[n:2*n]

  size_t m = t.size() + g.size() + h.size();

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
  size_t L_t = (t.size() + n_prime - 1) / n_prime;
  for (size_t i = 0; i < t.size(); i++) {
    s_prime.push_back(t[i]);
  }
  for (size_t i = t.size(); i < L_t * n_prime; i++) {
    s_prime.push_back(Rq());
  }

  // add the polynomials in g to s_prime and zero pad to make them L_g*n_prime length
  size_t L_g = (g.size() + n_prime - 1) / n_prime;
  for (size_t i = 0; i < g.size(); i++) {
    s_prime.push_back(g[i]);
  }
  for (size_t i = g.size(); i < L_g * n_prime; i++) {
    s_prime.push_back(Rq());
  }

  // add the polynomials in h to s_prime and zero pad to make them L_h*n_prime length
  size_t L_h = (h.size() + n_prime - 1) / n_prime;
  for (size_t i = 0; i < h.size(); i++) {
    s_prime.push_back(h[i]);
  }
  for (size_t i = h.size(); i < L_h * n_prime; i++) {
    s_prime.push_back(Rq());
  }

  // Step 7: Let recursive_instance be a new empty LabradorInstance
  size_t r_prime = 2 * nu + L_t + L_g + L_h;
  assert(r_prime == (s_prime.size() / n_prime));
  LabradorInstance recursive_instance(r_prime, n_prime, 10 * beta);

  // Step 8: add the equality constraint u1=Bt + Cg to recursive_instance
  // Generate B, C
  // TODO: change this so that B,C need not be computed and stored
  size_t l1 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base1));
  size_t l2 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base2));
  std::vector<Tq> B(kappa1 * l1 * r * kappa), C(kappa1 * ((r * (r + 1)) / 2) * l2);

  std::vector<std::byte> seed_B(ajtai_seed), seed_C(ajtai_seed);
  seed_B.push_back(std::byte('1'));
  seed_C.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(seed_B.data(), seed_B.size(), false, {}, B.data(), B.size()));
  ICICLE_CHECK(random_sampling(seed_C.data(), seed_C.size(), false, {}, C.data(), C.size()));

  assert(t.size() == l1 * r * kappa);
  assert(g.size() == ((r * (r + 1)) / 2) * l2);

  for (size_t i = 0; i < kappa1; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);
    size_t j = 0;
    while ((j + 1) * n_prime <= t.size()) {
      // new_constraint.phi[2*nu+j] = B[i][j*n_prime: (j+1)*n_prime]
      new_constraint.phi[2 * nu + j].assign(&B[i * t.size() + j * n_prime], &B[i * t.size() + (j + 1) * n_prime]);
      j++;
    }
    for (size_t k = 0; k < t.size() - j * n_prime; k++) {
      new_constraint.phi[2 * nu + j][k] = B[i * t.size() + j * n_prime + k];
    }

    j = 0;
    while ((j + 1) * n_prime <= g.size()) {
      // new_constraint.phi[2*nu + L_t + j] = C[i][j*n_prime: (j+1)*n_prime]
      new_constraint.phi[2 * nu + L_t + j].assign(&C[i * g.size() + j * n_prime], &C[i * g.size() + (j + 1) * n_prime]);
      j++;
    }
    for (size_t k = 0; k < g.size() - j * n_prime; k++) {
      new_constraint.phi[2 * nu + j][k] = B[i * l1 * r * kappa + j * n_prime + k];
    }
    new_constraint.b = u1[i];

    recursive_instance.add_equality_constraint(new_constraint);
  }

  // Step 9: add the equality constraint u2=Dh to recursive_instance
  // Generate D
  // TODO: change this so that D need not be computed and stored
  size_t l3 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base3));
  std::vector<Tq> D(kappa2 * l3 * ((r * (r + 1)) / 2));

  std::vector<std::byte> seed_D(ajtai_seed);
  seed_D.push_back(std::byte('3'));
  ICICLE_CHECK(random_sampling(seed_D.data(), seed_D.size(), false, {}, D.data(), D.size()));

  assert(h.size() == l3 * ((r * (r + 1)) / 2));

  for (size_t i = 0; i < kappa2; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);
    size_t j = 0;
    while ((j + 1) * n_prime <= h.size()) {
      // new_constraint.phi[2*nu + L_t +L_g +j] = D[i][j*n_prime: (j+1)*n_prime]
      new_constraint.phi[2 * nu + L_t + L_g + j].assign(
        &D[i * h.size() + j * n_prime], &D[i * h.size() + (j + 1) * n_prime]);
      j++;
    }
    for (size_t k = 0; k < h.size() - j * n_prime; k++) {
      new_constraint.phi[2 * nu + L_t + L_g + j][k] = D[i * h.size() + j * n_prime + k];
    }

    new_constraint.b = u2[i];

    recursive_instance.add_equality_constraint(new_constraint);
  }

  // Step 10: add the equality constraint Az - sum_i c_i t_i =0 to recursive_instance
  // Generate A
  // TODO: change this so that A need not be computed and stored
  std::vector<Tq> A(n * kappa);

  std::vector<std::byte> seed_A(ajtai_seed);
  seed_A.push_back(std::byte('0'));
  ICICLE_CHECK(random_sampling(seed_A.data(), seed_A.size(), false, {}, A.data(), n * kappa));

  for (size_t i = 0; i < kappa; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);
    size_t j = 0;
    while ((j + 1) * n_prime <= n) {
      // new_constraint.phi[j] = A[i][j*n_prime: (j+1)*n_prime]
      // new_constraint.phi[nu+ j] = A[i][j*n_prime: (j+1)*n_prime]
      new_constraint.phi[j].assign(&A[i * n + j * n_prime], &A[i * n + (j + 1) * n_prime]);
      new_constraint.phi[nu + j].assign(&A[i * n + j * n_prime], &A[i * n + (j + 1) * n_prime]);
      j++;
    }
    for (size_t k = 0; k < n - j * n_prime; k++) {
      new_constraint.phi[j][k] = A[i * n + j * n_prime + k];
      new_constraint.phi[nu + j][k] = A[i * n + j * n_prime + k];
    }

    // new_constraint.phi[nu+ j] = base0*new_constraint.phi[nu+ j]
    for (size_t k1 = 0; k1 < nu; k1++) {
      for (size_t k2 = 0; k2 < n_prime; k2++) {
        Zq base0_scalar = Zq::from(base0);
        ICICLE_CHECK(scalar_mul_vec(
          &base0_scalar, new_constraint.phi[nu + k1][k2].values, d, {}, new_constraint.phi[nu + k1][k2].values));
      }
    }

    // Step 10.d
    size_t k1 = 2 * nu, k2 = 0;
    for (size_t i1 = 0; i1 < r; i1++) {
      for (size_t i2 = 0; i2 < n; i2++) {
        for (size_t i3 = 0; i3 < l1; i3++) {
          if (i2 == i) {
            Tq temp;
            Zq base1_pow = Zq::from(static_cast<int64_t>(std::pow(base1, i3)));
            ICICLE_CHECK(scalar_mul_vec(&base1_pow, challenges_hat[i2].values, d, {}, temp.values));
            new_constraint.phi[k1][k2] = temp;
          }
          k2++;
          if (k2 == n_prime) {
            k2 = 0;
            k1++;
          }
        }
      }
    }

    recursive_instance.add_equality_constraint(new_constraint);
  }

  // Step 11:
  EqualityInstance step11_constraint(r_prime, n_prime);
  std::vector<Tq> c_times_phi(n);
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < n; j++) {
      Tq temp;
      ICICLE_CHECK(vector_mul(challenges_hat[i].values, final_const.phi[i][j].values, d, {}, temp.values));
      ICICLE_CHECK(vector_add(c_times_phi[j].values, temp.values, d, {}, c_times_phi[j].values));
    }
  }
  size_t idx = 0;
  while ((idx + 1) * n_prime <= n) {
    // step11_constraint.phi[j] = c_times_phi[j*n_prime: (j+1)*n_prime]
    // step11_constraint.phi[nu+ j] = c_times_phi[j*n_prime: (j+1)*n_prime]
    step11_constraint.phi[idx].assign(&c_times_phi[idx * n_prime], &c_times_phi[(idx + 1) * n_prime]);
    step11_constraint.phi[nu + idx].assign(&c_times_phi[idx * n_prime], &c_times_phi[(idx + 1) * n_prime]);
    idx++;
  }
  for (size_t k = 0; k < n - idx * n_prime; k++) {
    step11_constraint.phi[idx][k] = c_times_phi[idx * n_prime + k];
    step11_constraint.phi[nu + idx][k] = c_times_phi[idx * n_prime + k];
  }

  // step11_constraint.phi[nu+ j] = base0*step11_constraint.phi[nu+ j]
  for (size_t k1 = 0; k1 < nu; k1++) {
    for (size_t k2 = 0; k2 < n_prime; k2++) {
      Zq base0_scalar = Zq::from(base0);
      ICICLE_CHECK(scalar_mul_vec(
        &base0_scalar, step11_constraint.phi[nu + k1][k2].values, d, {}, step11_constraint.phi[nu + k1][k2].values));
    }
  }

  size_t s11_k1 = 2 * nu + L_t + L_g, s11_k2 = 0;
  for (size_t i1 = 0; i1 < r; i1++) {
    for (size_t i2 = i1; i2 < r; i2++) {
      for (size_t i3 = 0; i3 < l3; i3++) {
        Tq temp;
        ICICLE_CHECK(vector_mul(challenges_hat[i1].values, challenges_hat[i2].values, d, {}, temp.values));
        Zq multiplier = Zq::from(-1 * std::pow(base3, i3));
        if (i1 != i2) { multiplier = Zq::from(2) * multiplier; }
        ICICLE_CHECK(scalar_mul_vec(&multiplier, temp.values, d, {}, temp.values));
        step11_constraint.phi[s11_k1][s11_k2] = temp;
        s11_k2++;
        if (s11_k2 == n_prime) {
          s11_k2 = 0;
          s11_k1++;
        }
      }
    }
  }
  recursive_instance.add_equality_constraint(step11_constraint);

  // Step 12:
  EqualityInstance step12_constraint(r_prime, n_prime);

  size_t s12_k1 = 2 * nu + L_t, s12_k2 = 0;
  for (size_t i1 = 0; i1 < r; i1++) {
    for (size_t i2 = i1; i2 < r; i2++) {
      for (size_t i3 = 0; i3 < l2; i3++) {
        Tq temp = final_const.a[i1][i2];
        Zq multiplier = Zq::from(std::pow(base2, i3));
        if (i1 != i2) { multiplier = Zq::from(2) * multiplier; }

        ICICLE_CHECK(scalar_mul_vec(&multiplier, temp.values, d, {}, temp.values));
        step12_constraint.phi[s12_k1][s12_k2] = temp;
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
          Zq multiplier = Zq::from(std::pow(base3, i3));
          for (size_t k = 0; k < d; k++) {
            temp.values[k] = multiplier;
          }
          step12_constraint.phi[s12_k1][s12_k2] = temp;
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
  recursive_instance.add_equality_constraint(step12_constraint);

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
      step13_constraint.a[i1][i2] = temp;
    }
  }

  size_t s13_k1 = 2 * nu + L_t, s13_k2 = 0;
  for (size_t i1 = 0; i1 < r; i1++) {
    for (size_t i2 = i1; i2 < r; i2++) {
      for (size_t i3 = 0; i3 < l2; i3++) {
        Tq temp;
        ICICLE_CHECK(vector_mul(challenges_hat[i1].values, challenges_hat[i2].values, d, {}, temp.values));
        Zq multiplier = Zq::from(-1 * std::pow(base2, i3));
        if (i1 != i2) { multiplier = Zq::from(2) * multiplier; }
        ICICLE_CHECK(scalar_mul_vec(&multiplier, temp.values, d, {}, temp.values));
        step11_constraint.phi[s13_k1][s13_k2] = temp;
        s13_k2++;
        if (s13_k2 == n_prime) {
          s13_k2 = 0;
          s13_k1++;
        }
      }
    }
  }
  recursive_instance.add_equality_constraint(step11_constraint);

  // Step 14: set new beta
  // TODO: what should the value be?
  recursive_instance.beta = 100.0;
  return std::make_pair(recursive_instance, s_prime);
}

void prover(LabradorInstance lab_inst, const std::vector<Rq>& S, size_t num_rec)
{
  std::vector<std::byte> ajtai_seed;
  const char* str = "INSERT PREFIXED RANDOM VALUE HERE";
  ajtai_seed.assign(reinterpret_cast<const std::byte*>(str), reinterpret_cast<const std::byte*>(str + strlen(str)));
  std::vector<Zq> proof;
  std::vector<Rq> witness(S);
  LabradorProtocol L;
  for (size_t i = 0; i < num_rec; i++) {
    ajtai_seed.push_back(std::byte(i));
    LabradorRecursionRawInstance raw_inst = L.base_prover(lab_inst, ajtai_seed, witness, proof);
    auto [new_lab_inst, new_witness] = L.prepare_recursive_problem(ajtai_seed, raw_inst, 1 << 4, 1 << 4);
    lab_inst = new_lab_inst;
    witness = new_witness;
    ajtai_seed.pop_back();
  }
}

eIcicleError verify(/*TODO params*/)
{
  // TODO Ash: labrador verifier
  return eIcicleError::SUCCESS;
}

// === Main driver ===

int main(int argc, char* argv[])
{
  ICICLE_LOG_INFO << "Labrador example";
  try_load_and_set_backend_device(argc, argv);

  const int64_t q = get_q<Zq>();

  // TODO use icicle_malloc() instead of std::vector. Consider a DeviceVector<T> that behaves like std::vector

  // randomize the witness Si with low norm
  const size_t n = 1 << 8;
  const size_t r = 1 << 8;
  constexpr size_t d = Rq::d;
  std::vector<Rq> S(r * n);

  // TODO: generate dot-product constraints and a witness that solve them for the proof to be valid
  auto randomize_Rq_vec = [](std::vector<Rq>& vec, int64_t max_value) {
    for (auto& x : vec) {
      for (size_t i = 0; i < d; ++i) {                    // randomize each coefficient
        uint64_t val = rand_uint_32b() % (max_value + 1); // uniform in [0, sqrt_q]
        x.values[i] = Zq::from(val);
      }
    }
  };

  // generate random values in [0, sqrt(q)]. We assume witness is low norm.
  const int64_t sqrt_q = static_cast<int64_t>(std::sqrt(q));
  randomize_Rq_vec(S, sqrt_q);

  // === Call the protocol ===
  // ICICLE_CHECK(prove(/* TODO(Ash): add arguments */));
  // ICICLE_CHECK(verify(/* TODO(Ash): add arguments */));

  return 0;
}