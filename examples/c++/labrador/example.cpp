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

/// extracts the symmetric part of a n X n matrix as a n(n+1)/2 size vector
template <typename T>
std::vector<T> extract_symm_part(T* mat, size_t n)
{
  size_t n_choose_2 = (n * (n + 1)) / 2;
  std::vector<T> v(n_choose_2);
  size_t offset = 0;

  // Create stream for async operations
  icicleStreamHandle stream;
  icicle_create_stream(&stream);

  for (size_t i = 0; i < n; i++) {
    icicle_copy_async(&v[offset], &mat[i * n + i], (n - i) * sizeof(T), stream);
    offset += n - i;
  }

  // Synchronize to ensure all copies complete before returning
  icicle_stream_synchronize(stream);
  icicle_destroy_stream(stream);

  return v;
}

std::pair<size_t, std::vector<Zq>> LabradorBaseProver::select_valid_jl_proj(std::byte* seed, size_t seed_len) const
{
  size_t JL_out = lab_inst.param.JL_out;
  size_t n = lab_inst.n;
  size_t r = lab_inst.r;
  size_t d = Rq::d;

  std::vector<Zq> p(JL_out, Zq::from(0));
  size_t JL_i = 0;
  // TODO:convert this to just 1 call of JL projection
  while (true) {
    std::vector<std::byte> base_jl_seed(seed, seed + seed_len);
    base_jl_seed.push_back(std::byte(JL_i));

    for (size_t j = 0; j < r; j++) {
      // add byte j to the seed
      std::vector<std::byte> jl_seed(base_jl_seed);
      jl_seed.push_back(std::byte(j));

      std::vector<Zq> s_projection(JL_out);
      // create JL projection: P_j*s_j
      ICICLE_CHECK(jl_projection(
        reinterpret_cast<const Zq*>(S.data() + j * n), n * d, jl_seed.data(), jl_seed.size(), {}, s_projection.data(),
        JL_out));
      // add output to p
      vector_add(p.data(), s_projection.data(), s_projection.size(), {}, p.data());
    }
    // check norm
    bool JL_check = false;
    double beta = lab_inst.param.beta;
    ICICLE_CHECK(check_norm_bound(p.data(), JL_out, eNormType::L2, uint64_t(sqrt(JL_out / 2) * beta), {}, &JL_check));

    if (JL_check) {
      break;
    } else {
      p.assign(p.size(), Zq::from(0));
      JL_i++;
    }
  }
  // at the end JL projection is defined by JL_i and p is the projection output
  // return these
  return std::make_pair(JL_i, p);
}

std::vector<Rq> compute_Q_poly(size_t n, size_t r, size_t JL_out, std::byte* seed, size_t seed_len, size_t JL_i)
{
  size_t d = Rq::d;
  // Step 17: Create conjugated polynomial vectors from JL matrix rows
  std::vector<Rq> Q(r * JL_out * n);
  for (size_t i = 0; i < r; i++) {
    // Create seed for P_i matrix (same as in step 12)
    std::vector<std::byte> base_jl_seed(seed, seed + seed_len);
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
  return Q;
}

std::vector<Rq> sample_low_norm_challenges(size_t n, size_t r, std::byte* seed, size_t seed_len)
{
  size_t d = Rq::d;
  std::vector<Rq> challenge(r);
  std::vector<size_t> j_ch(r, 0);
  // TODO: can parallelise the i loop
  for (size_t i = 0; i < r; i++) {
    while (true) {
      std::vector<std::byte> ch_seed(seed, seed + seed_len);
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
  return challenge;
}

// modifies the instance
// returns num_aggregation_rounds number of polynomials
std::vector<Tq> LabradorInstance::agg_const_zero_constraints(
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
          Tq a_ij_l = const_zero_constraints[l].a[i * r + j];

          // Scalar multiply and add: sum += psi_scalar * a_ij_l
          // TODO: use vector_mul<Rq,Zq> and vector_sum<Rq> to aggregate in a vectorized way
          Tq temp;
          ICICLE_CHECK(scalar_mul_vec(&psi_scalar, a_ij_l.values, d, {}, temp.values));
          ICICLE_CHECK(vector_add(sum.values, temp.values, d, {}, sum.values));
        }

        new_constraint.a[r * i + j] = sum;
      }
    }

    // Compute varphi'_i^{(k)} = sum_{l=0}^{L-1} psi^{(k)}(l) * phi'_i^{(l)} + sum_{l=0}^{255} omega^{(k)}(l) * q_{il}
    for (size_t i = 0; i < r; i++) {
      // First sum: over const_zero_constraints
      for (size_t l = 0; l < L; l++) {
        Zq psi_scalar = psi[psi_index(k, l)];

        for (size_t m = 0; m < n; m++) {
          Tq phi_il_m = const_zero_constraints[l].phi[i * n + m];

          // phi_prime[i,m] += psi_scalar * phi_il_m
          Tq temp;
          ICICLE_CHECK(scalar_mul_vec(&psi_scalar, phi_il_m.values, d, {}, temp.values));
          ICICLE_CHECK(
            vector_add(new_constraint.phi[i * n + m].values, temp.values, d, {}, new_constraint.phi[i * n + m].values));
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
            vector_add(new_constraint.phi[i * n + m].values, temp.values, d, {}, new_constraint.phi[i * n + m].values));
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
          a_vec.push_back(new_constraint.a[i * r + i]);
        } else {
          // Off-diagonal: (a''_{ij} + a''_{ji})
          Tq a_ij = new_constraint.a[i * r + j];
          Tq a_ji = new_constraint.a[j * r + i];
          Tq temp;
          ICICLE_CHECK(vector_add(a_ij.values, a_ji.values, d, {}, temp.values));
          a_vec.push_back(temp);
        }
      }
    }

    ICICLE_CHECK(matmul(g_hat.data(), 1, g_hat.size(), a_vec.data(), a_vec.size(), 1, {}, &new_constraint.b));

    // Second part: sum_i <phi'_i^{(k)}, s_i>

    Tq prod;
    ICICLE_CHECK(matmul(new_constraint.phi.data(), 1, r * n, S_hat.data(), r * n, 1, {}, &prod));
    ICICLE_CHECK(vector_add(new_constraint.b.values, prod.values, d, {}, new_constraint.b.values));

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

void LabradorInstance::agg_equality_constraints(const std::vector<Tq>& alpha_hat)
{ // Step 22: Say the EqualityInstances in LabradorInstance are:
  // [{a_{ij}^{(k)}; 0 ≤ i,j < r} ⊂ T_q, b^{(k)} ∈ T_q, {φ_i^{(k)} : 0 ≤ i < r} ⊂ T_q^n : 0 ≤ k < K]

  // For 0 ≤ i,j < r, the Prover computes a''_{ij}:
  const size_t K = equality_constraints.size();
  const size_t d = Rq::d;
  std::vector<Tq> a_final(r * n, zero());
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < r; j++) {
      // a''_{ij} = ∑_{k=0}^{K-1} α_k * a_{ij}^{(k)} (multiplication in T_q)
      for (size_t k = 0; k < K; k++) {
        // Get a_{ij}^{(k)} from equality constraint k (already in T_q)
        Tq a_ij_k = equality_constraints[k].a[i * r + j];

        // Multiply by α_k and add to sum (T_q operations)
        Tq temp;
        ICICLE_CHECK(vector_mul(alpha_hat[k].values, a_ij_k.values, d, {}, temp.values));
        ICICLE_CHECK(vector_add(a_final[i * r + j].values, temp.values, d, {}, a_final[i * r + j].values));
      }
    }
  }

  // For 0 ≤ i < r, the Prover computes φ'_i:
  std::vector<Tq> phi_final(r * n, zero());
  for (size_t i = 0; i < r; i++) {
    for (size_t m = 0; m < n; m++) {
      // φ'_i[m] = ∑_{k=0}^{K-1} α_k * φ_i^{(k)}[m] (multiplication in T_q)
      for (size_t k = 0; k < K; k++) {
        // Get φ_i^{(k)}[m] from equality constraint k (already in T_q)
        Tq phi_i_k_m = equality_constraints[k].phi[i * n + m];

        // Multiply by α_k and add to sum (T_q operations)
        Tq temp;
        ICICLE_CHECK(vector_mul(alpha_hat[k].values, phi_i_k_m.values, d, {}, temp.values));
        ICICLE_CHECK(vector_add(phi_final[i * n + m].values, temp.values, d, {}, phi_final[i * n + m].values));
      }
    }
  }

  // The Prover also computes b':
  Tq b_final;

  for (size_t k = 0; k < K; k++) {
    // Get b^{(k)} from equality constraint k (already in T_q)
    Tq b_k = equality_constraints[k].b;

    // Multiply by α_k and add to sum (T_q operations)
    Tq temp;
    ICICLE_CHECK(vector_mul(alpha_hat[k].values, b_k.values, d, {}, temp.values));
    ICICLE_CHECK(vector_add(b_final.values, temp.values, d, {}, b_final.values));
  }

  // roll a_final, phi_final, b_final into a single EqualityInstance and put it in the equality_constraints
  EqualityInstance final_const(r, n, a_final, phi_final, b_final);

  equality_constraints = {final_const};
}

// This destroys the lab_inst in LabradorBaseProver
std::pair<LabradorBaseCaseProof, PartialTranscript> LabradorBaseProver::base_case_prover()
{
  // Step 1: Pack the Witnesses into a Matrix S
  const size_t r = lab_inst.r; // Number of witness vectors
  const size_t n = lab_inst.n; // Dimension of witness vectors
  constexpr size_t d = Rq::d;
  // Ensure S is of the correct size
  if (S.size() != r * n) { throw std::invalid_argument("S must have size r * n"); }

  PartialTranscript trs;
  // Step 2: Convert S to the NTT Domain
  std::vector<Tq> S_hat(r * n);
  // Perform negacyclic NTT on the witness S
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

  // Step 3: S@A = T
  const std::vector<std::byte>& ajtai_seed = lab_inst.param.ajtai_seed;
  std::vector<std::byte> seed_A(ajtai_seed);
  seed_A.push_back(std::byte('0'));

  // Use ajtai_commitment to compute T_hat = S_hat @ A
  size_t kappa = lab_inst.param.kappa;
  Tq* T_hat_ptr = ajtai_commitment(seed_A.data(), seed_A.size(), n, kappa, S_hat.data(), r * n);
  std::vector<Tq> T_hat(T_hat_ptr, T_hat_ptr + r * kappa);

  // Step 4: already done

  // Step 5: Convert T_hat to Rq
  std::vector<Rq> T(r * kappa);
  // Perform negacyclic INTT
  ICICLE_CHECK(ntt(T_hat.data(), r * kappa, NTTDir::kInverse, {}, T.data()));

  // Step 6: decompose T to T_tilde
  size_t base1 = lab_inst.param.base1;
  size_t l1 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base1);
  std::vector<Rq> T_tilde(l1 * r * kappa);
  ICICLE_CHECK(decompose(T.data(), r * kappa, base1, {}, T_tilde.data(), T_tilde.size()));

  // Step 7: compute g
  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  std::vector<Tq> g_hat = extract_symm_part(G_hat.data(), r);
  size_t r_choose_2 = (r * (r + 1)) / 2;
  std::vector<Rq> g(r_choose_2);

  ICICLE_CHECK(ntt(g_hat.data(), r_choose_2, NTTDir::kInverse, {}, g.data()));

  // Step 8: decompose g to g_tilde
  size_t base2 = lab_inst.param.base2;
  size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base2);
  std::vector<Rq> g_tilde(l2 * g.size());
  ICICLE_CHECK(decompose(g.data(), g.size(), base2, {}, g_tilde.data(), g_tilde.size()));

  // Step 9: already done

  // Step 10: u1 = B@T_tilde + C@g_tilde
  // Generate B, C
  // TODO: change this so that B,C need not be computed and stored
  size_t kappa1 = lab_inst.param.kappa1;
  std::vector<std::byte> seed_B(ajtai_seed), seed_C(ajtai_seed);
  seed_B.push_back(std::byte('1'));
  seed_C.push_back(std::byte('2'));

  // compute NTTs for T_tilde, g_tilde
  std::vector<Tq> T_tilde_hat(T_tilde.size()), g_tilde_hat(g_tilde.size());
  ICICLE_CHECK(ntt(T_tilde.data(), T_tilde.size(), NTTDir::kForward, {}, T_tilde_hat.data()));
  ICICLE_CHECK(ntt(g_tilde.data(), g_tilde.size(), NTTDir::kForward, {}, g_tilde_hat.data()));

  // v1 = B@T_tilde
  Tq* v1_ptr =
    ajtai_commitment(seed_B.data(), seed_B.size(), l1 * r * kappa, kappa1, T_tilde_hat.data(), T_tilde_hat.size());
  std::vector<Tq> v1(v1_ptr, v1_ptr + kappa1);
  // v2 = C@g_tilde
  Tq* v2_ptr =
    ajtai_commitment(seed_C.data(), seed_C.size(), (r_choose_2)*l2, kappa1, g_tilde_hat.data(), g_tilde_hat.size());
  std::vector<Tq> v2(v2_ptr, v2_ptr + kappa1);

  std::vector<Tq> u1(kappa1);
  vector_add(v1.data(), v2.data(), kappa1, {}, u1.data());

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

  // Step 12: Select a JL projection
  size_t JL_out = lab_inst.param.JL_out;
  auto [JL_i, p] = select_valid_jl_proj(seed1.data(), seed1.size());

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

  // Step 14: removed
  // Step 15, 16: already done

  // Step 17: Create conjugated polynomial vectors from JL matrix rows
  std::vector<Rq> Q = compute_Q_poly(n, r, JL_out, seed1.data(), seed1.size(), JL_i);
  // indexes into a multidim array of dim = r X JL_out X n
  auto Q_index = [n, JL_out](size_t i, size_t j, size_t k) { return (i * JL_out * n + j * n + k); };

  // Step 18: Let L be the number of constZeroInstance constraints in LabradorInstance.
  // For 0 ≤ k < ceil(128/log(q)), sample the following random vectors:
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t num_aggregation_rounds = std::ceil(128.0 / std::log2(get_q<Zq>()));

  std::vector<Zq> psi(num_aggregation_rounds * L), omega(num_aggregation_rounds * JL_out);
  // indexes into multidim arrays: psi[k][l] and omega[k][l]
  auto psi_index = [L](size_t k, size_t l) { return k * L + l; };
  auto omega_index = [JL_out](size_t k, size_t l) { return k * JL_out + l; };

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
  // Step 19: Aggregate ConstZeroInstance constraints
  // For every 0 ≤ k < ceil(128/log(q)) compute aggregated constraints

  std::vector<Tq> msg3 =
    lab_inst.agg_const_zero_constraints(num_aggregation_rounds, JL_out, S_hat, g_hat, Q, psi, omega);

  // Step 20: seed3 = hash(seed2, msg3)
  // TODO: add serialization to msg3 and put them in the placeholder
  std::vector<std::byte> seed3(hasher.output_size());
  hasher.hash("Placeholder3", 12, {}, seed3.data());

  trs.b_agg = msg3;
  trs.seed3 = seed3;
  // Step 21: Sample random polynomial vectors α using seed3
  // Let K be the number of EqualityInstances in the LabradorInstance
  const size_t K = lab_inst.equality_constraints.size();

  std::vector<Tq> alpha_hat(K);
  std::vector<std::byte> alpha_seed(seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(alpha_seed.data(), alpha_seed.size(), false, {}, alpha_hat.data(), K));

  trs.alpha_hat = alpha_hat;
  // Step 22:
  lab_inst.agg_equality_constraints(alpha_hat);

  // Step 23: For 0 ≤ i ≤ j < r, the Prover computes the matrix multiplication between matrix
  // Phi = (φ'_0|φ'_1|···|φ'_{r-1})^T ∈ R_q^{r×n} and S ∈ R_q^{r×n} defined earlier.
  // Let H ∈ R_q^{r×r}, such that H = 2^{-1}(Phi @ S^T + (Phi @ S^T)^T)

  // Matrix Phi
  const Tq* phi_final = lab_inst.equality_constraints[0].phi.data();

  // Compute Phi @ S^T using the transposed S_hat
  std::vector<Tq> Phi_times_St_hat(r * r);
  ICICLE_CHECK(matmul(phi_final, r, n, S_hat_transposed.data(), n, r, {}, Phi_times_St_hat.data()));

  // Convert back to Rq domain
  std::vector<Rq> Phi_times_St(r * r), Phi_times_St_transposed(r * r);
  ICICLE_CHECK(ntt(Phi_times_St_hat.data(), r * r, NTTDir::kInverse, {}, Phi_times_St.data()));
  // transpose matrix
  ICICLE_CHECK(matrix_transpose<Tq>(Phi_times_St.data(), r, r, {}, Phi_times_St_transposed.data()));

  // Compute H = 2^{-1}(LS + (LS)^T)
  Zq two_inv = Zq::inverse(Zq::from(2)); // 2^{-1} in Z_q

  // Phi_times_St = Phi_times_St + Phi_times_St_transposed
  ICICLE_CHECK(vector_add(Phi_times_St.data(), Phi_times_St_transposed.data(), r * r, {}, Phi_times_St.data()));
  // Phi_times_St = 1/2 * Phi_times_St
  ICICLE_CHECK(scalar_mul_vec(
    &two_inv, reinterpret_cast<Zq*>(Phi_times_St.data()), r * r, {}, reinterpret_cast<Zq*>(Phi_times_St.data())));

  std::vector<Rq> H = extract_symm_part(Phi_times_St.data(), r);

  // Step 24: Decompose h
  size_t base3 = lab_inst.param.base3;
  size_t l3 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base3);

  std::vector<Rq> H_tilde(l3 * H.size());
  ICICLE_CHECK(decompose(H.data(), H.size(), base3, {}, H_tilde.data(), H_tilde.size()));
  std::vector<Tq> H_tilde_hat(H_tilde.size());
  ICICLE_CHECK(ntt(H_tilde.data(), H_tilde.size(), NTTDir::kForward, {}, H_tilde_hat.data()));

  // Step 25: already done
  // Step 26: commit to H_tilde
  size_t kappa2 = lab_inst.param.kappa2;
  std::vector<std::byte> seed_D(ajtai_seed);
  seed_D.push_back(std::byte('3'));

  // u2 = D@H_tilde
  Tq* u2_ptr =
    ajtai_commitment(seed_D.data(), seed_D.size(), l3 * r_choose_2, kappa2, H_tilde_hat.data(), H_tilde_hat.size());
  std::vector<Tq> u2(u2_ptr, u2_ptr + kappa2);
  // Step 27:
  // add u2 to the trs
  trs.u2 = u2;

  // TODO: add serialization to u2 and put them in the placeholder
  std::vector<std::byte> seed4(hasher.output_size());
  hasher.hash("Placeholder4", 12, {}, seed4.data());

  trs.seed4 = seed4;
  // Step 28: sampling low operator norm challenges
  std::vector<Rq> challenge = sample_low_norm_challenges(n, r, seed4.data(), seed4.size());

  std::vector<Tq> challenges_hat(r);
  ICICLE_CHECK(ntt(challenge.data(), challenge.size(), NTTDir::kForward, {}, challenges_hat.data()));
  trs.challenges_hat = challenges_hat;

  // Step 29: Compute z_hat
  std::vector<Tq> z_hat(n);
  // TODO: vectorise this
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < r; j++) {
      Tq temp;
      ICICLE_CHECK(vector_mul(challenges_hat[j].values, S_hat[j * n + i].values, d, {}, temp.values));
      ICICLE_CHECK(vector_add(z_hat[i].values, temp.values, d, {}, z_hat[i].values));
    }
  }
  LabradorBaseCaseProof final_proof{lab_inst.equality_constraints[0], z_hat, T_tilde, g_tilde, H_tilde};

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

    std::copy(&B_t[i * t_len], &B_t[(i + 1) * t_len], &new_constraint.phi[2 * nu * n_prime]);
    std::copy(&C_t[i * g_len], &C_t[(i + 1) * g_len], &new_constraint.phi[(2 * nu + L_t) * n_prime]);

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

    std::copy(&D_t[i * h_len], &D_t[(i + 1) * h_len], &new_constraint.phi[(2 * nu + L_t + L_g) * n_prime]);
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

    std::copy(&A_t[i * n], &A_t[(i + 1) * n], new_constraint.phi.data());
    std::copy(&A_t[i * n], &A_t[(i + 1) * n], new_constraint.phi[nu * n_prime]);
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
  std::copy(c_times_phi.begin(), c_times_phi.end(), step11_constraint.phi.data());
  std::copy(c_times_phi.begin(), c_times_phi.end(), &step11_constraint.phi[nu]);

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
        Zq multiplier = Zq::from(-1 * std::pow(prev_param.base3, i3));
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
        Zq multiplier = Zq::from(-1 * std::pow(prev_param.base2, i3));
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

// eIcicleError verify(/*TODO params*/)
// {
//   // TODO Ash: labrador verifier
//   return eIcicleError::SUCCESS;
// }

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

// n*r = 2^30
// r^2 = n
// r= 2^10, n=2^20

// A*B
// A: k X n
// B: n X r

// Zq, base = q^{1/t} t=2,4,6
