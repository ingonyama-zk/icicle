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
  std::vector<std::vector<Rq>> a;   // a[i][j] matrix over Rq (r x r matrix)
  std::vector<std::vector<Rq>> phi; // phi[i] vector over Rq (r vectors, each of size n)
  Rq b;                             // Polynomial in Rq

  EqualityInstance(size_t r, size_t n) : r(r), n(n), a(r, std::vector<Rq>(r)), phi(r, std::vector<Rq>(n)), b() {}
};

struct ConstZeroInstance {
  const size_t r;                   // Number of witness vectors
  const size_t n;                   // Dimension of each vector in Rq
  std::vector<std::vector<Rq>> a;   // a[i][j] matrix over Rq (r x r matrix)
  std::vector<std::vector<Rq>> phi; // phi[i] vector over Rq (r vectors, each of size n)
  Rq b;                             // Polynomial in Rq

  ConstZeroInstance(size_t r, size_t n) : r(r), n(n), a(r, std::vector<Rq>(r)), phi(r, std::vector<Rq>(n)), b() {}
};

struct LabradorInstance {
  const size_t r;                                        // Number of witness vectors
  const size_t n;                                        // Dimension of each vector in Rq
  double beta;                                           // Norm bound
  std::vector<EqualityInstance> equality_constraints;    // K EqualityInstances
  std::vector<ConstZeroInstance> const_zero_constraints; // L ConstZeroInstances

  LabradorInstance(size_t r, size_t n, double beta) : r(r), n(n), beta(beta) {}

  // Add an EqualityInstance
  void add_equality_constraint(const EqualityInstance& instance) { equality_constraints.push_back(instance); }

  // Add a ConstZeroInstance
  void add_const_zero_constraint(const ConstZeroInstance& instance) { const_zero_constraints.push_back(instance); }
};

Rq conjugate(const Rq& p)
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
  const LabradorInstance lab_inst,
  const std::vector<std::byte> ajtai_seed,
  const std::vector<Rq> S,
  std::vector<Zq> proof)
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
  size_t l1 = std::log2(get_q<Zq>()) / std::log2(base1) + 1;
  std::vector<Rq> T_tilde(l1 * r * kappa);
  ICICLE_CHECK(decompose(T.data(), r * kappa, base1, {}, T_tilde.data(), T_tilde.size()));

  // Step 7: compute g
  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  std::vector<Rq> g;
  for (size_t i = 0; i < r; i++) {
    for (size_t j = i; j < r; j++) {
      Rq temp;
      ICICLE_CHECK(ntt(G_hat[i * r + j].coeffs, d, NTTDir::kInverse, default_ntt_config<Zq>(), temp.coeffs));
      g.push_back(temp);
    }
  }

  // Step 8: Convert g to g_tilde
  uint32_t base2 = 1 << 16;
  size_t l2 = std::log2(get_q<Zq>()) / std::log2(base2) + 1;
  std::vector<Rq> g_tilde(l2 * g.size());
  ICICLE_CHECK(decompose(g.data(), g.size(), base2, {}, g_tilde.data(), g_tilde.size()));

  // Step 9: already done
  // vector(t) = T_tilde
  // vector(g) = g_tilde

  // Step 10: u1 = B@T_tilde + C@g_tilde
  // Generate B, C
  // TODO: change this so that B,C need not be computed and stored
  const size_t kappa1 = 1 << 4;
  std::vector<Tq> B(kappa1 * l1 * r * kappa), C(kappa1 * r * (r + 1) * l2 / 2);

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
    matmul(C.data(), kappa1, r * (r + 1) * l2 / 2, g_tilde_ntt.data(), r * (r + 1) * l2 / 2, 1, {}, v2.data()));
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
  std::vector<Rq> r_ijk(r * JL_out * n), q_ijk(r * JL_out * n);
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

      // Convert R_ij to polynomial vector r_ijk[i,j,:] ∈ R_q^n
      for (size_t k = 0; k < n; k++) {
        // Use R_ij[k*d:(k+1)*d] as coefficients for polynomial k
        for (size_t coeff = 0; coeff < d; coeff++) {
          r_ijk[rq_index(i, j, k)].coeffs[coeff] = R_ij[k * d + coeff];
        }

        // Define q_ijk[k] = conjugation(r_ijk[k])
        q_ijk[rq_index(i, j, k)] = conjugate(r_ijk[rq_index(i, j, k)]);
      }
    }
  }

  // Step 18: Let L be the number of constZeroInstance constraints in LabradorInstance.
  // For 0 ≤ k < ceil(128/log(q)), sample the following random vectors:
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t num_aggregation_rounds = static_cast<size_t>(std::ceil(128.0 / std::log2(get_q<Zq>())));

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

  return eIcicleError::SUCCESS;
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