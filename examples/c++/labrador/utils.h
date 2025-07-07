#pragma once

#include "icicle/runtime.h"
#include "labrador.h"

using namespace icicle::labrador;

/// returns 0 polynomial
PolyRing zero();

/// returns q for Zq
template <typename Zq>
int64_t get_q()
{
  constexpr auto q_storage = Zq::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  return q;
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

/// reconstructs a symmetric n × n matrix from its packed upper–triangular
/// representation produced by extract_symm_part.
/// The returned matrix is in row-major order and satisfies
///     v == extract_symm_part(M.data(), n)
template <typename T>
std::vector<T> reconstruct_symm_matrix(const std::vector<T>& v, size_t n)
{
  // Make sure the caller supplied the right-sized vector
  assert(v.size() == (n * (n + 1)) / 2 && "Packed vector has wrong length");

  std::vector<T> mat(n * n);

  // Copy the packed upper-triangular part row by row.
  icicleStreamHandle stream;
  icicle_create_stream(&stream);

  size_t offset = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t len = n - i; // -- elements on and right of the diagonal
    icicle_copy_async(
      &mat[i * n + i], // destination start  (row i, col i)
      &v[offset],      // source start
      len * sizeof(T), // bytes to copy
      stream);
    offset += len;
  }

  // Wait for the DMA copies to complete before we mirror the data
  icicle_stream_synchronize(stream);
  icicle_destroy_stream(stream);

  // Mirror the upper-triangular entries to the lower-triangular part
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      mat[j * n + i] = mat[i * n + j];
    }
  }

  return mat;
}

// ----------- Generic helper (works for Zq etc.) ------------------
/// computes the trace of a n X n matrix and stores the result in trace_result
template <typename T>
eIcicleError compute_matrix_trace(const T* matrix, size_t n, T* trace_result)
{
  // 1. grab the diagonal
  std::vector<T> diagonal(n);
  eIcicleError err = slice(
    matrix,
    /*offset */ 0,
    /*stride*/ n + 1,
    /*size_in */ n * n,
    /*size_out*/ n, {}, diagonal.data());
  if (err != eIcicleError::SUCCESS) return err;

  // 2. sum it
  return vector_sum(diagonal.data(), n, {}, trace_result);
}

/// Computes the trace of a polynomial matrix
/// PolyRing overload – treats the n×n polynomial matrix as an
/// (n×n)*d Zq-matrix and re-uses slice / vector_sum on Zq.
inline eIcicleError compute_matrix_trace(const PolyRing* matrix, size_t n, PolyRing* trace_result)
{
  constexpr size_t d = PolyRing::d;

  const Zq* flat = reinterpret_cast<const Zq*>(matrix);
  Zq* out = reinterpret_cast<Zq*>(trace_result);

  const size_t stride = (n + 1) * d; // jump between diagonal coefficients
  const size_t size_in = n * n * d;  // total #Zq words

  std::vector<Zq> diag_coeff(d * n);

  VecOpsConfig async_config = default_vec_ops_config();
  async_config.is_async = true;

  for (size_t coeff = 0; coeff < d; ++coeff) {
    // collect the `coeff`-th coefficient from every diagonal polynomial
    eIcicleError err = slice(
      flat,
      /*offset */ coeff,
      /*stride*/ stride, size_in, n, async_config, &diag_coeff[coeff * n]);
    if (err != eIcicleError::SUCCESS) return err;
  }

  VecOpsConfig sum_config = default_vec_ops_config();
  sum_config.batch_size = d; // d independent vectors of length n
  eIcicleError err = vector_sum(diag_coeff.data(), n, sum_config, out);

  if (err != eIcicleError::SUCCESS) return err;
  return eIcicleError::SUCCESS;
}