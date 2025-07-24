#pragma once

#include "labrador.h"           // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h" // For Hash

#include "types.h"
#include "utils.h"
#include "oracle.h"

using namespace icicle::labrador;

/// @brief Compares two vectors of PolyRing polynomials element-wise
/// @param vec1 First vector to compare
/// @param vec2 Second vector to compare
/// @param size Number of PolyRing elements to compare
/// @return true if vectors are equal, false otherwise
bool poly_vec_eq(const PolyRing* vec1, const PolyRing* vec2, size_t size);

/// @brief Computes the Ajtai commitment of the given input S. Views input S as matrix of vectors to be committed.
/// Vectors are arranged in the row major form. If A is the Ajtai matrix, then this outputs S@A.
/// @param A Random Ajtai commitment matrix
/// @param input_len length of vectors to be committed
/// @param output_len length of commitments
/// @param S data to be committed
/// @param S_len length of data to be committed. If `S_len > input_len` then S_len must be a multiple of input_len. The
/// input S will be viewed as a row major arrangement of S_len/input_len vectors to be committed.
/// @return S_len/input_len commitments of length equal to output_len arranged in row major form.
std::vector<Tq>
ajtai_commitment(const std::vector<Tq>& A, size_t input_len, size_t output_len, const Tq* S, size_t S_len);

/// Samples low norm challenge polynomials for the Labrador protocol
std::vector<Rq> sample_low_norm_challenges(size_t n, size_t r, const std::byte* seed, size_t seed_len);

/// Helper to concatenate oracle_seed and lab_inst bytes and return an Oracle object.
Oracle create_oracle_seed(const std::byte* seed, size_t seed_len, const LabradorInstance& inst);

/// Returns the LabradorInstance for recursion problem
LabradorInstance prepare_recursion_instance(
  const LabradorParam& prev_param,
  const EqualityInstance& final_const,
  const PartialTranscript& trs,
  uint32_t base0,
  size_t mu,
  size_t nu);

/// @brief Helper struct to prepare recursive instance of the problem
struct RecursionPreparer {
  size_t prev_r, prev_n, mu, nu;
  /// decomposition basis for z
  uint32_t base0;

  // These are internally created
  size_t n_prime, r_prime;
  size_t t_len, g_len, h_len;
  size_t L_t, L_g, L_h;

  RecursionPreparer(const LabradorParam& param, size_t mu, size_t nu, uint32_t base0)
      : prev_r(param.r), prev_n(param.n), mu(mu), nu(nu), base0(base0)
  {
    t_len = param.t_len();
    g_len = param.g_len();
    h_len = param.h_len();
    size_t m = t_len + g_len + h_len;

    n_prime = std::max(std::ceil((double)prev_n / nu), std::ceil((double)m / mu));

    L_t = (t_len + n_prime - 1) / n_prime;
    L_g = (g_len + n_prime - 1) / n_prime;
    L_h = (h_len + n_prime - 1) / n_prime;
    r_prime = (2 * nu + L_t + L_g + L_h);
  }

  // These functions return the starting index for corresponding witness in a r_prime * n_prime matrix
  size_t z0_begin_idx() const;
  size_t z1_begin_idx() const;
  size_t t_begin_idx() const;
  size_t g_begin_idx() const;
  size_t h_begin_idx() const;

  // NOTE: for these functions need to ensure that the size of dst is r_prime * n_prime and src has correct size (same
  // as z0, z1, t, g, h depending on what is being called)

  // Needs dst size = r_prime * n_prime and src size = prev_n
  eIcicleError copy_like_z0(Rq* dst, const Rq* src) const;
  // Needs dst size = r_prime * n_prime and src size = prev_n
  eIcicleError copy_like_z1(Rq* dst, const Rq* src) const;
  // Needs dst size = r_prime * n_prime and src size = t_len
  eIcicleError copy_like_t(Rq* dst, const Rq* src) const;
  // Needs dst size = r_prime * n_prime and src size = g_len
  eIcicleError copy_like_g(Rq* dst, const Rq* src) const;
  // Needs dst size = r_prime * n_prime and src size = h_len
  eIcicleError copy_like_h(Rq* dst, const Rq* src) const;
};

/// choose base0 such that at the end of a base_case_prover witness z satisfies
/// ||z||_2 <op_norm_bound * beta * sqrt(r) < base0^2
uint32_t calc_base0(size_t r, uint64_t op_norm_bound, double beta);

/// returns a choice of mu, nu given n, m for recursion
std::pair<size_t, size_t> compute_mu_nu(size_t n, size_t m);

/// return kappa such that for a random A: kappa X m the MSIS problem
/// Ax = 0 has at least 128 bits of security
size_t secure_msis_rank();