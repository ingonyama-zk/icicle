#include "examples_utils.h"

// ICICLE runtime
#include "icicle/runtime.h"

#include "icicle/lattice/labrador.h" // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h"      // For Hash

using namespace icicle::labrador;

constexpr size_t beta = 10; // TODO(Ash): set beta according to the protocol

// === TODO(Ash): Consider adding protocol-specific types ===
struct EqualityInstance {
  const size_t r;                   // Number of witness vectors
  const size_t n;                   // Dimension of each vector in Rq
  std::vector<std::vector<Tq>> a;   // a[i][j] matrix over Rq (r x r matrix)
  std::vector<std::vector<Tq>> phi; // phi[i] vector over Rq (r vectors, each of size n)
  Tq b;                             // Polynomial in Rq

  EqualityInstance(size_t r, size_t n) : r(r), n(n), a(r, std::vector<Tq>(r)), phi(r, std::vector<Tq>(n)), b() {}
  EqualityInstance(size_t r, size_t n, std::vector<std::vector<Tq>> a, std::vector<std::vector<Tq>> phi, Tq b)
      : r(r), n(n), a(std::move(a)), phi(std::move(phi)), b(std::move(b))
  {
    // check if the sizes of a and phi are correct
    if (a.size() != r || phi.size() != r) {
      throw std::invalid_argument("EqualityInstance: 'a' and 'phi' must have size r");
    }
    for (const auto& row : a) {
      if (row.size() != r) { throw std::invalid_argument("EqualityInstance: each row of 'a' must have size r"); }
    }
    for (const auto& vec : phi) {
      if (vec.size() != n) { throw std::invalid_argument("EqualityInstance: each vector in 'phi' must have size n"); }
    }
  }
};

struct ConstZeroInstance {
  const size_t r;                   // Number of witness vectors
  const size_t n;                   // Dimension of each vector in Rq
  std::vector<std::vector<Tq>> a;   // a[i][j] matrix over Tq (r x r matrix)
  std::vector<std::vector<Tq>> phi; // phi[i] vector over Tq (r vectors, each of size n)
  Tq b;                             // Polynomial in Rq

  ConstZeroInstance(size_t r, size_t n) : r(r), n(n), a(r, std::vector<Tq>(r)), phi(r, std::vector<Tq>(n)), b() {}
};

struct LabradorInstance {
  const size_t r;                                        // Number of witness vectors
  const size_t n;                                        // Dimension of each vector in Tq
  double beta;                                           // Norm bound
  std::vector<EqualityInstance> equality_constraints;    // K EqualityInstances
  std::vector<ConstZeroInstance> const_zero_constraints; // L ConstZeroInstances

  LabradorInstance(size_t r, size_t n, double beta) : r(r), n(n), beta(beta) {}

  // Add an EqualityInstance
  void add_equality_constraint(const EqualityInstance& instance) { equality_constraints.push_back(instance); }

  // Add a ConstZeroInstance
  void add_const_zero_constraint(const ConstZeroInstance& instance) { const_zero_constraints.push_back(instance); }
};

// TODO: add a LabradorProver struct which contains all the relevant parameters

Rq icicle::labrador::conjugate(const Rq& p)
{
  Rq result;
  // Copy constant term (index 0)
  result.coeffs[0] = p.coeffs[0];

  // For remaining coefficients, flip and negate
  for (size_t k = 1; k < Rq::d; k++) {
    // TODO: neg is negate?
    result.coeffs[k] = Zq::neg(p.coeffs[Rq::d - k]);
  }

  return result;
}

template <typename Zq>
int64_t get_q()
{
  constexpr auto q_storage = Zq::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  return q;
}

// === TODO(Ash): Implement protocol logic ===

eIcicleError setup(/*TODO params*/)
{
  // TODO Ash: labrador setup
  return eIcicleError::SUCCESS;
}

eIcicleError base_prover(
  LabradorInstance lab_inst, const std::vector<std::byte> ajtai_seed, const std::vector<Rq> S, std::vector<Zq> proof)
{
  // Step 1: Pack the Witnesses into a Matrix S

  const size_t r = lab_inst.r; // Number of witness vectors
  const size_t n = lab_inst.n; // Dimension of witness vectors
  constexpr size_t d = Rq::d;
  // Ensure S is of the correct size
  if (S.size() != r * n) { return eIcicleError::INVALID_ARGUMENT; }

  // Setup negacyclic NTT config for Zq
  // TODO: not sure how this code gets modified
  const unsigned log_ntt_size = d + 1;
  Zq basic_root = Zq::omega(log_ntt_size);
  auto ntt_init_domain_cfg = default_ntt_init_domain_config();
  ICICLE_CHECK(ntt_init_domain(basic_root, ntt_init_domain_cfg));

  // // ntt configuration
  // NTTConfig<Zq> ntt_cfg = default_ntt_config<Zq>();
  // // ConfigExtension ntt_cfg_ext;
  // // config.ext = &ntt_cfg_ext;
  // // config.batch_size = batch_size;

  // Step 2: Convert S to the NTT Domain
  std::vector<Tq> S_hat(r * n);

  for (size_t i = 0; i < r * n; ++i) {
    // Perform negacyclic NTT
    ICICLE_CHECK(ntt(S[i].coeffs, d, NTTDir::kForward, default_ntt_config<Zq>(), S_hat[i].coeffs));
  }

  // Step 3: S@A = T
  // Generate A
  // TODO: change this so that A need not be computed and stored
  const size_t kappa = 1 << 4;
  std::vector<Tq> A(n * kappa);

  std::vector<std::byte> seed_A(ajtai_seed);
  seed_A.push_back(std::byte('0'));
  ICICLE_CHECK(random_sampling<Tq>(seed_A.data(), seed_A.size(), false, {}, A.data(), n * kappa));

  std::vector<Tq> T_hat(r * kappa);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, A.data(), n, kappa, {}, T_hat.data()));

  // Step 4: already done

  // Step 5: Convert T_hat to Rq
  std::vector<Rq> T(r * kappa);

  for (size_t i = 0; i < r * kappa; ++i) {
    // Perform negacyclic INTT
    ICICLE_CHECK(ntt(T_hat[i].coeffs, d, NTTDir::kInverse, default_ntt_config<Zq>(), T[i].coeffs));
  }

  // Step 6: Convert T to T_tilde
  uint32_t base1 = 1 << 16;
  size_t l1 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base1));
  std::vector<Rq> T_tilde(l1 * r * kappa);
  ICICLE_CHECK(decompose(T.data(), r * kappa, base1, {}, T_tilde.data(), T_tilde.size()));

  // Step 7: compute g
  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  std::vector<Rq> g;
  std::vector<Tq> g_hat;
  for (size_t i = 0; i < r; i++) {
    for (size_t j = i; j < r; j++) {
      g_hat.push_back(G_hat[i * r + j]);
      Rq temp;
      ICICLE_CHECK(ntt(G_hat[i * r + j].coeffs, d, NTTDir::kInverse, default_ntt_config<Zq>(), temp.coeffs));
      g.push_back(temp);
    }
  }

  // Step 8: Convert g to g_tilde
  uint32_t base2 = 1 << 16;
  size_t l2 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base2));
  std::vector<Rq> g_tilde(l2 * g.size());
  ICICLE_CHECK(decompose(g.data(), g.size(), base2, {}, g_tilde.data(), g_tilde.size()));

  // Step 9: already done
  // vector(t) = T_tilde
  // vector(g) = g_tilde

  // Step 10: u1 = B@T_tilde + C@g_tilde
  // Generate B, C
  // TODO: change this so that B,C need not be computed and stored
  const size_t kappa1 = 1 << 4;
  std::vector<Tq> B(kappa1 * l1 * r * kappa), C(kappa1 * ((r * (r + 1)) / 2) * l2);

  std::vector<std::byte> seed_B(ajtai_seed), seed_C(ajtai_seed);
  seed_B.push_back(std::byte('1'));
  seed_C.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling<Tq>(seed_B.data(), seed_B.size(), false, {}, B.data(), B.size()));
  ICICLE_CHECK(random_sampling<Tq>(seed_C.data(), seed_C.size(), false, {}, C.data(), C.size()));

  // compute NTTs for T_tilde, g_tilde
  std::vector<Tq> T_tilde_ntt(T_tilde.size()), g_tilde_ntt(g_tilde.size());
  for (size_t i = 0; i < T_tilde.size(); ++i) {
    ICICLE_CHECK(ntt(T_tilde[i].coeffs, d, NTTDir::kForward, default_ntt_config<Zq>(), T_tilde_ntt[i].coeffs));
  }
  for (size_t i = 0; i < g_tilde.size(); ++i) {
    ICICLE_CHECK(ntt(g_tilde[i].coeffs, d, NTTDir::kForward, default_ntt_config<Zq>(), g_tilde_ntt[i].coeffs));
  }

  std::vector<Tq> u1(kappa1), v1(kappa1), v2(kappa1);
  // v1 = B@T_tilde
  ICICLE_CHECK(matmul(B.data(), kappa1, l1 * r * kappa, T_tilde_ntt.data(), l1 * r * kappa, 1, {}, v1.data()));
  // v2 = C@g_tilde
  ICICLE_CHECK(
    matmul(C.data(), kappa1, ((r * (r + 1)) / 2) * l2, g_tilde_ntt.data(), ((r * (r + 1)) / 2) * l2, 1, {}, v2.data()));
  for (size_t i = 0; i < kappa1; i++) {
    // TODO: can we flatten v1, v2 as Zq and run this?
    vector_add(v1[i].coeffs, v2[i].coeffs, d, {}, u1[i].coeffs);
  }

  // Step 11: hash (lab_inst, ajtai_seed, u1) to get seed1
  // add u1 to the proof
  for (size_t i = 0; i < kappa1; i++) {
    proof.insert(proof.end(), u1[i].coeffs, u1[i].coeffs + d);
  }
  // hash and get a challenge
  Hash hasher = create_sha3_256_hash();
  // TODO: add serialization to lab_inst, ajtai_seed, u1 and put them in the placeholder
  std::vector<std::byte> seed1(hasher.output_size());
  hasher.hash("Placeholder1", 12, {}, seed1.data());

  // Step 12: Select a JL projection
  constexpr size_t JL_out = 256;
  std::vector<Zq> p(JL_out, Zq::from(0));
  size_t JL_i = 0;
  while (true) {
    std::vector<std::byte> base_jl_seed(seed1);
    base_jl_seed.push_back(std::byte(JL_i));

    for (size_t j = 0; j < r; j++) {
      // add byte j to the seed
      std::vector<std::byte> jl_seed(base_jl_seed);
      jl_seed.push_back(std::byte(j));

      std::vector<Zq> input, output(JL_out);
      // unpack row k of S into input
      for (size_t k = 0; k < n; k++) {
        input.insert(input.end(), S[j * n + k].coeffs, S[j * n + k].coeffs + d);
      }
      // create JL projection: P_j*s_j
      ICICLE_CHECK(
        jl_projection(input.data(), input.size(), jl_seed.data(), jl_seed.size(), {}, output.data(), JL_out));
      // add output to p
      vector_add(p.data(), output.data(), JL_out, {}, p.data());
    }
    // check norm
    bool JL_check = false;
    ICICLE_CHECK(check_norm_bound(p.data(), JL_out, eNormType::L2, uint64_t(sqrt(128) * beta), {}, &JL_check));

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
  hasher.hash("Placeholder2", 12, {}, seed2.data());

  // Step 14: removed
  // Step 15, 16: already done

  // Step 17: Create polynomial vectors from JL matrix rows
  std::vector<Rq> R(r * JL_out * n), Q(r * JL_out * n);
  // indexes into a multidim array of dim = r X JL_out X n
  auto rq_index = [n](size_t i, size_t j, size_t k) { return (i * JL_out * n + j * n + k); };

  for (size_t i = 0; i < r; i++) {
    // Create seed for P_i matrix (same as in step 12)
    std::vector<std::byte> base_jl_seed(seed1);
    base_jl_seed.push_back(std::byte(JL_i));
    base_jl_seed.push_back(std::byte(i));

    for (size_t j = 0; j < JL_out; j++) {
      // Get row j of matrix P_i
      std::vector<Zq> R_ij(n * d);
      ICICLE_CHECK(get_jl_matrix_row(
        base_jl_seed.data(), base_jl_seed.size(),
        JL_out, // matrix_rows (N = 256)
        n * d,  // matrix_cols (M = n*d)
        j,      // row_index
        {},     // config
        R_ij.data()));

      // Convert R_ij to polynomial vector R[i,j,:] ∈ R_q^n
      for (size_t k = 0; k < n; k++) {
        // Use R_ij[k*d:(k+1)*d] as coefficients for polynomial k
        for (size_t coeff = 0; coeff < d; coeff++) {
          R[rq_index(i, j, k)].coeffs[coeff] = R_ij[k * d + coeff];
        }

        // Define Q[k] = conjugation(R[k])
        Q[rq_index(i, j, k)] = conjugate(R[rq_index(i, j, k)]);
      }
    }
  }

  // Step 18: Let L be the number of constZeroInstance constraints in LabradorInstance.
  // For 0 ≤ k < ceil(128/log(q)), sample the following random vectors:
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t num_aggregation_rounds = std::ceil(128.0 / std::log2(get_q<Zq>()));

  std::vector<Zq> psi_k(num_aggregation_rounds * L), omega_k(num_aggregation_rounds * JL_out);
  // indexes into multidim arrays: psi[k][l] and omega[k][l]
  auto psi_index = [L](size_t k, size_t l) { return k * L + l; };
  auto omega_index = [](size_t k, size_t l) { return k * JL_out + l; };

  // sample psi_k
  std::vector<std::byte> psi_seed(seed2);
  psi_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling<Zq>(psi_seed.data(), psi_seed.size(), false, {}, psi_k.data(), psi_k.size()));

  // Sample omega_k
  std::vector<std::byte> omega_seed(seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling<Zq>(omega_seed.data(), omega_seed.size(), false, {}, omega_k.data(), omega_k.size()));

  // Step 19: Aggregate ConstZeroInstance constraints
  // For every 0 ≤ k < ceil(128/log(q)) compute aggregated constraints

  std::vector<Zq> msg3;
  for (size_t k = 0; k < num_aggregation_rounds; k++) {
    EqualityInstance new_constraint(r, n);

    // Compute a''_{ij} = sum_{l=0}^{L-1} psi^{(k)}(l) * a'_{ij}^{(l)}
    for (size_t i = 0; i < r; i++) {
      for (size_t j = 0; j < r; j++) {
        Tq sum = Tq(); // Initialize to zero polynomial

        for (size_t l = 0; l < L; l++) {
          // Get psi^{(k)}(l) as scalar
          Zq psi_scalar = psi_k[psi_index(k, l)];

          // Get a_{ij}^{(l)} from const_zero_constraints
          Tq a_ij_l = lab_inst.const_zero_constraints[l].a[i][j];

          // Scalar multiply and add: sum += psi_scalar * a_ij_l
          Tq temp;
          ICICLE_CHECK(scalar_mul_vec(&psi_scalar, a_ij_l.coeffs, d, {}, temp.coeffs));
          ICICLE_CHECK(vector_add(sum.coeffs, temp.coeffs, d, {}, sum.coeffs));
        }

        new_constraint.a[i][j] = sum;
      }
    }

    // Compute varphi'_i^{(k)} = sum_{l=0}^{L-1} psi^{(k)}(l) * phi'_i^{(l)} + sum_{l=0}^{255} omega^{(k)}(l) * q_{il}
    for (size_t i = 0; i < r; i++) {
      // First sum: over const_zero_constraints
      for (size_t l = 0; l < L; l++) {
        Zq psi_scalar = psi_k[psi_index(k, l)];

        for (size_t m = 0; m < n; m++) {
          Tq phi_il_m = lab_inst.const_zero_constraints[l].phi[i][m];

          // phi_prime[i,m] += psi_scalar * phi_il_m
          Tq temp;
          ICICLE_CHECK(scalar_mul_vec(&psi_scalar, phi_il_m.coeffs, d, {}, temp.coeffs));
          ICICLE_CHECK(
            vector_add(new_constraint.phi[i][m].coeffs, temp.coeffs, d, {}, new_constraint.phi[i][m].coeffs));
        }
      }

      // Second sum: over JL projection rows (256 rows)
      for (size_t l = 0; l < JL_out; l++) { // JL_out = 256
        Zq omega_scalar = omega_k[omega_index(k, l)];

        for (size_t m = 0; m < n; m++) {
          // q_{il} is stored in Q[i][l][:]
          Rq q_ilm = Q[rq_index(i, l, m)];
          Tq q_ilm_hat;
          ntt(q_ilm.coeffs, d, NTTDir::kForward, {}, q_ilm_hat.coeffs);

          // phi_i_k[m] += omega_scalar * q_ilm
          Tq temp;
          ICICLE_CHECK(scalar_mul_vec(&omega_scalar, q_ilm_hat.coeffs, d, {}, temp.coeffs));
          ICICLE_CHECK(
            vector_add(new_constraint.phi[i][m].coeffs, temp.coeffs, d, {}, new_constraint.phi[i][m].coeffs));
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
          ICICLE_CHECK(vector_add(a_ij.coeffs, a_ji.coeffs, d, {}, temp.coeffs));
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
      ICICLE_CHECK(vector_add(new_constraint.b.coeffs, prod.coeffs, d, {}, new_constraint.b.coeffs));
    }

    // Add the EqualityInstance to LabradorInstance
    lab_inst.add_equality_constraint(new_constraint);

    // Send B^(k) to the Verifier
    msg3.insert(msg3.end(), new_constraint.b.coeffs, new_constraint.b.coeffs + d);
  }

  // Step 20: seed3 = hash(seed2, msg3)
  // TODO: add serialization to msg3 and put them in the placeholder
  std::vector<std::byte> seed3(hasher.output_size());
  hasher.hash("Placeholder3", 12, {}, seed3.data());

  proof.insert(proof.end(), msg3.begin(), msg3.end());

  // Step 21: Sample random polynomial vectors α using seed3
  // Let K be the number of EqualityInstances in the LabradorInstance
  const size_t K = lab_inst.equality_constraints.size();

  std::vector<Tq> alpha_hat(K);
  std::vector<std::byte> alpha_seed(seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling<Tq>(alpha_seed.data(), alpha_seed.size(), false, {}, alpha_hat.data(), K));

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
        ICICLE_CHECK(vector_mul(alpha_hat[k].coeffs, a_ij_k.coeffs, d, {}, temp.coeffs));
        ICICLE_CHECK(vector_add(a_final[i][j].coeffs, temp.coeffs, d, {}, a_final[i][j].coeffs));
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
        ICICLE_CHECK(vector_mul(alpha_hat[k].coeffs, phi_i_k_m.coeffs, d, {}, temp.coeffs));
        ICICLE_CHECK(vector_add(phi_final[i][m].coeffs, temp.coeffs, d, {}, phi_final[i][m].coeffs));
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
    ICICLE_CHECK(vector_mul(alpha_hat[k].coeffs, b_k.coeffs, d, {}, temp.coeffs));
    ICICLE_CHECK(vector_add(b_final.coeffs, temp.coeffs, d, {}, b_final.coeffs));
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
  for (size_t i = 0; i < r * r; i++) {
    ICICLE_CHECK(ntt(LS_hat[i].coeffs, d, NTTDir::kInverse, default_ntt_config<Zq>(), LS[i].coeffs));
  }

  // Compute H = 2^{-1}(LS + LS^T)
  std::vector<Rq> H;
  Zq two_inv = Zq::inverse(Zq::from(2)); // 2^{-1} in Z_q

  for (size_t i = 0; i < r; i++) {
    // only upper triangular elements
    for (size_t j = i; j < r; j++) {
      // H[i][j] = 2^{-1} * (LS[i][j] + LS[j][i])
      Rq temp;
      ICICLE_CHECK(vector_add(LS[i * r + j].coeffs, LS[j * r + i].coeffs, d, {}, temp.coeffs));
      ICICLE_CHECK(scalar_mul_vec(&two_inv, temp.coeffs, d, {}, temp.coeffs));
      H.push_back(temp);
    }
  }

  // Step 24: Decompose h
  uint32_t base3 = 1 << 16; // Choose appropriate base
  size_t l3 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base3));

  std::vector<Rq> H_tilde(l3 * H.size());
  ICICLE_CHECK(decompose(H.data(), H.size(), base3, {}, H_tilde.data(), H_tilde.size()));
  std::vector<Tq> H_tilde_ntt;
  for (const auto& poly : H_tilde) {
    Tq temp;
    ICICLE_CHECK(ntt(poly.coeffs, d, NTTDir::kForward, default_ntt_config<Zq>(), temp.coeffs));
    H_tilde_ntt.push_back(temp);
  }

  // Step 25: already done
  // Step 26: commit to H_tilde
  constexpr size_t kappa2 = 1 << 4;
  std::vector<Tq> D(kappa2 * l3 * ((r * (r + 1)) / 2));

  std::vector<Tq> u2(kappa2);
  // u2 = D@H_tilde
  ICICLE_CHECK(
    matmul(D.data(), kappa2, l3 * ((r * (r + 1)) / 2), H_tilde_ntt.data(), H_tilde_ntt.size(), 1, {}, u2.data()));

  // Step 27:
  // add u2 to the proof
  for (size_t i = 0; i < kappa2; i++) {
    proof.insert(proof.end(), u2[i].coeffs, u2[i].coeffs + d);
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
      ICICLE_CHECK(check_norm_bound(challenge[i].coeffs, d, eNormType::Lop, 15, {}, &norm_bound));

      if (norm_bound) {
        break;
      } else {
        j_ch[i]++;
      }
    }
  }

  std::vector<Tq> challenge_hat(r);
  for (size_t i = 0; i < r; i++) {
    ICICLE_CHECK(ntt(challenge[i].coeffs, d, NTTDir::kForward, {}, challenge_hat[i].coeffs));
  }

  // Step 29: Compute z_hat
  std::vector<Tq> z_hat(n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < r; j++) {
      Tq temp;
      ICICLE_CHECK(vector_mul(challenge_hat[j].coeffs, S_hat[j * n + i].coeffs, d, {}, temp.coeffs));
      ICICLE_CHECK(vector_add(z_hat[i].coeffs, temp.coeffs, d, {}, z_hat[i].coeffs));
    }
  }
  return eIcicleError::SUCCESS;
}

std::pair<LabradorInstance, std::vector<Rq>> prepare_recursive_problem(
  EqualityInstance final_const,
  std::vector<std::byte> ajtai_seed,
  std::vector<Tq> challenges_hat,
  std::vector<Tq> z_hat,
  std::vector<Tq> t,
  std::vector<Tq> g,
  std::vector<Tq> h)
{
  const size_t r = final_const.r;
  const size_t n = final_const.n;
  constexpr size_t d = Rq::d;

  // Step 1: Convert z_hat back to polynomial domain
  std::vector<Rq> z(n);
  for (size_t i = 0; i < n; i++) {
    ICICLE_CHECK(ntt(z_hat[i].coeffs, d, NTTDir::kInverse, default_ntt_config<Zq>(), z[i].coeffs));
  }

  // Step 2: Decompose z using base b_0
  uint32_t base0 = 1 << 16; // Choose appropriate base (same pattern as base protocol)
  size_t l0 = std::ceil(std::log2(get_q<Zq>()) / std::log2(base0));

  // Decompose all elements first
  std::vector<Rq> z_decomposed_full(l0 * n);
  ICICLE_CHECK(decompose(z.data(), n, base0, {}, z_decomposed_full.data(), z_decomposed_full.size()));

  // Take only first 2n elements (in NTT domain)- all the rest should be 0
  std::vector<Tq> z_tilde(2 * n);
  for (size_t i = 0; i < 2 * n && i < z_decomposed_full.size(); i++) {
    ICICLE_CHECK(ntt(z_decomposed_full[i].coeffs, d, NTTDir::kForward, default_ntt_config<Zq>(), z_tilde[i].coeffs));
  }

  // Step 3:
  // z0 = z_tilde[:n]
  // z1 = z_tilde[n:2*n]

  size_t m = t.size() + g.size() + h.size();

  // Step 4, 5:
  size_t nu = 1 << 3;
  size_t mu = 1 << 3;
  size_t n_prime = std::max(std::ceil((double)n / nu), std::ceil((double)m / mu));

  // Step 6
  // we will view s_prime as a multidimensional array. At the base level it consists of n_prime length vectors
  std::vector<Tq> s_prime;

  for (size_t i = 0; i < n; i++) {
    s_prime.push_back(z_tilde[i]);
  }
  for (size_t i = n; i < nu * n_prime; i++) {
    s_prime.push_back(Tq());
  }
  // now s_prime is nu*n_prime long and can be viewed as a nu long array of n_prime dimension Tq vectors

  for (size_t i = 0; i < n; i++) {
    s_prime.push_back(z_tilde[n + i]);
  }
  for (size_t i = n; i < nu * n_prime; i++) {
    s_prime.push_back(Tq());
  }
  // now s_prime is 2*nu*n_prime long and can be viewed as a 2*nu long array of n_prime dimension Tq vectors

  // add the polynomials in t to s_prime and zero pad to make them L_t*n_prime length
  size_t L_t = std::ceil((double)t.size() / n_prime);
  for (size_t i = 0; i < t.size(); i++) {
    s_prime.push_back(t[i]);
  }
  for (size_t i = t.size(); i < L_t * n_prime; i++) {
    s_prime.push_back(Tq());
  }

  // add the polynomials in g to s_prime and zero pad to make them L_g*n_prime length
  size_t L_g = std::ceil((double)g.size() / n_prime);
  for (size_t i = 0; i < g.size(); i++) {
    s_prime.push_back(g[i]);
  }
  for (size_t i = g.size(); i < L_g * n_prime; i++) {
    s_prime.push_back(Tq());
  }

  // add the polynomials in h to s_prime and zero pad to make them L_h*n_prime length
  size_t L_h = std::ceil((double)h.size() / n_prime);
  for (size_t i = 0; i < h.size(); i++) {
    s_prime.push_back(h[i]);
  }
  for (size_t i = h.size(); i < L_h * n_prime; i++) {
    s_prime.push_back(Tq());
  }

  LabradorInstance recursive_instance(0, 0, 0); // Placeholder
  std::vector<Rq> recursive_witness;            // Placeholder

  return std::make_pair(recursive_instance, recursive_witness);
}

eIcicleError verify(/*TODO params*/)
{
  // TODO Ash: labrador verifier
  return eIcicleError::SUCCESS;
}

// === Main driver ===

int main(int argc, char* argv[])
{
  try_load_and_set_backend_device(argc, argv);

  int64_t q = get_q<Zq>();

  // randomize the witness Si with low norm
  // TODO Ash: maybe want to allocate them consecutive in memory
  const size_t n = 1 << 8;
  const size_t r = 1 << 8;
  constexpr size_t d = Rq::d;
  std::vector<Rq> S(r * n);

  // TODO eventually we will use icicle_malloc() and icicle_copy() to allocate and copy that is device agnostic and
  // support GPU too. First step can be with host memory and then we can add device support.

  auto randomize_Rq_vec = [](std::vector<Rq>& vec, int64_t max_value) {
    for (auto& x : vec) {
      for (size_t i = 0; i < d; ++i) {                    // randomize each coefficient
        uint64_t val = rand_uint_32b() % (max_value + 1); // uniform in [0, sqrt_q]
        x.coeffs[i] = Zq::from(val);
      }
    }
  };

  // std::cout << "0= " << Zq::from(0) << std::endl
  //           << "1= " << Zq::from(1) << std::endl
  //           << "31= " << Zq::from(31) << std::endl;

  // generate random values in [0, sqrt(q)]. We assume witness is low norm.
  const int64_t sqrt_q = static_cast<int64_t>(std::sqrt(q));
  randomize_Rq_vec(S, sqrt_q);

  // === Call the protocol ===
  // ICICLE_CHECK(setup(/* TODO(Ash): add arguments */));
  // ICICLE_CHECK(prove(/* TODO(Ash): add arguments */));
  // ICICLE_CHECK(verify(/* TODO(Ash): add arguments */));

  std::cout << "Hello\n";
  return 0;
}