#pragma once

#include "icicle/lattice/labrador.h" // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h"      // For Hash

#include "types.h"
#include "utils.h"

using namespace icicle::labrador;

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
  const std::byte* ajtai_mat_seed, size_t seed_len, size_t input_len, size_t output_len, const Tq* S, size_t S_len);

/// returns Q: JL_out X r X n matrix such that
/// Q(i,:,:) is the conjugation of the ith row of the JL projection viewed as a polynomial vector.
/// So that const(<Q(i,:,:), S(:,:)>) = p_i
/// JL_i needs to be the same as the one given by select_valid_jl_proj
std::vector<Rq> compute_Q_poly(size_t n, size_t r, size_t JL_out, std::byte* seed, size_t seed_len, size_t JL_i);

// TODO: Simply returns the polynomial x for every challenge rn
std::vector<Rq> sample_low_norm_challenges(size_t n, size_t r, std::byte* seed, size_t seed_len);

/// Returns the LabradorInstance for recursion problem
LabradorInstance prepare_recursion_instance(
  const LabradorParam& prev_param,
  const EqualityInstance& final_const,
  const PartialTranscript& trs,
  size_t base0,
  size_t mu,
  size_t nu);