#pragma once

#include "icicle/sumcheck/sumcheck_proof.h"
#include "icicle/serialization.h"

namespace icicle {
  template <typename S>
  struct BinarySerializeImpl<SumcheckProof<S>> {
    static eIcicleError serialized_size(const SumcheckProof<S>& obj, size_t& size)
    {
      size = obj.get_nof_round_polynomials(); // nof_round_polynomials

      for (size_t i = 0; i < obj.get_nof_round_polynomials(); i++) {
        const auto& round_poly = obj.get_const_round_polynomial(i);
        size += sizeof(size_t); // nested vector length
        size += round_poly.size() * sizeof(S);
      }
      return eIcicleError::SUCCESS;
    }
    static eIcicleError pack_and_advance(std::byte*& buffer, size_t& buffer_length, const SumcheckProof<S>& obj)
    {
      size_t nof_round_polynomials = obj.get_nof_round_polynomials();
      ICICLE_RETURN_IF_ERR(memcpy_shift_destination(buffer, buffer_length, &nof_round_polynomials, sizeof(size_t)));
      for (size_t i = 0; i < nof_round_polynomials; i++) {
        const auto& round_poly = obj.get_const_round_polynomial(i);
        size_t round_poly_size = round_poly.size();
        ICICLE_RETURN_IF_ERR(memcpy_shift_destination(buffer, buffer_length, &round_poly_size, sizeof(size_t)));
        ICICLE_RETURN_IF_ERR(
          memcpy_shift_destination(buffer, buffer_length, round_poly.data(), round_poly_size * sizeof(S)));
      }
      return eIcicleError::SUCCESS;
    }
    static eIcicleError unpack_and_advance(const std::byte*& buffer, size_t& buffer_length, SumcheckProof<S>& obj)
    {
      size_t nof_round_polynomials;
      std::vector<std::vector<S>> round_polynomials;
      ICICLE_RETURN_IF_ERR(memcpy_shift_source(&nof_round_polynomials, buffer_length, buffer, sizeof(size_t)));

      round_polynomials.resize(nof_round_polynomials);
      for (size_t i = 0; i < nof_round_polynomials; ++i) {
        size_t round_poly_size;
        ICICLE_RETURN_IF_ERR(memcpy_shift_source(&round_poly_size, buffer_length, buffer, sizeof(size_t)));

        size_t byte_size = round_poly_size * sizeof(S);
        round_polynomials[i].resize(round_poly_size);
        ICICLE_RETURN_IF_ERR(memcpy_shift_source(round_polynomials[i].data(), buffer_length, buffer, byte_size));
      }

      SumcheckProof<S> proof = SumcheckProof<S>(std::move(round_polynomials));
      obj = std::move(proof);
      return eIcicleError::SUCCESS;
    }
  };
} // namespace icicle
