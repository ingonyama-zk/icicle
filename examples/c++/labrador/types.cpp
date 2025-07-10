#include "types.h"

// Modifies equality constraints
void LabradorInstance::agg_equality_constraints(const std::vector<Tq>& alpha_hat)
{ // Step 22: Say the EqualityInstances in LabradorInstance are:
  // [{a_{ij}^{(k)}; 0 ≤ i,j < r} ⊂ T_q, b^{(k)} ∈ T_q, {φ_i^{(k)} : 0 ≤ i < r} ⊂ T_q^n : 0 ≤ k < K]

  // For 0 ≤ i,j < r, the Prover computes a''_{ij}:
  const size_t K = equality_constraints.size();
  assert(K == alpha_hat.size() && "alpha_hat size insufficient");
  size_t r = param.r;
  size_t n = param.n;
  const size_t d = Tq::d;
  EqualityInstance final_const(param.r, param.n);

  MatMulConfig async_config{};
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