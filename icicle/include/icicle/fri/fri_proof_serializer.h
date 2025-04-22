#pragma once

#include "icicle/fri/fri_proof.h"
#include "icicle/merkle/merkle_proof_serializer.h"
#include "icicle/serialization.h"

namespace icicle {
  template <typename F>
  struct BinarySerializeImpl<FriProof<F>> {
    static eIcicleError serialized_size(const FriProof<F>& obj, size_t& size)
    {
      size = sizeof(size_t); // nof_queries
      size_t nof_queries = obj.get_nof_fri_queries();
      size_t nof_rounds = obj.get_nof_fri_rounds();
      for (size_t i = 0; i < nof_queries; i++) {
        size += sizeof(size_t); // nof_fri_rounds
        for (size_t j = 0; j < nof_rounds; j++) {
          const auto& proof = obj.get_query_proof_slot(i, j);
          size_t proof_size = 0;
          ICICLE_CHECK_IF_RETURN(BinarySerializeImpl<MerkleProof>::serialized_size(proof, proof_size));
          size += proof_size;
        }
      }
      size += sizeof(size_t); // final_poly_size
      size += obj.get_final_poly_size() * sizeof(F);
      size += sizeof(uint64_t); // pow_nonce

      return eIcicleError::SUCCESS;
    }
    static eIcicleError pack_and_advance(std::byte*& buffer, size_t& buffer_length, const FriProof<F>& obj)
    {
      size_t query_proofs_size = obj.get_nof_fri_queries();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &query_proofs_size, sizeof(size_t)));
      for (size_t i = 0; i < query_proofs_size; i++) {
        size_t nof_rounds = obj.get_nof_fri_rounds();
        ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &nof_rounds, sizeof(size_t)));
        for (size_t j = 0; j < nof_rounds; j++) {
          const auto& proof = obj.get_query_proof_slot(i, j);
          ICICLE_CHECK_IF_RETURN(BinarySerializeImpl<MerkleProof>::pack_and_advance(buffer, buffer_length, proof));
        }
      }
      size_t final_poly_size = obj.get_final_poly_size();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &final_poly_size, sizeof(size_t)));
      ICICLE_CHECK_IF_RETURN(
        memcpy_shift_destination(buffer, buffer_length, obj.get_final_poly(), final_poly_size * sizeof(F)));
      uint64_t pow_nonce = obj.get_pow_nonce();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &pow_nonce, sizeof(uint64_t)));
      return eIcicleError::SUCCESS;
    }
    static eIcicleError unpack_and_advance(const std::byte*& buffer, size_t& buffer_length, FriProof<F>& obj)
    {
      size_t min_required_length =
        sizeof(size_t) + sizeof(size_t) + sizeof(size_t) + sizeof(uint64_t); // minimum length of the proof
      if (buffer_length < min_required_length) {
        ICICLE_LOG_ERROR << "Deserialization failed: buffer_length < min_required_length: " << buffer_length << " < "
                         << min_required_length;
        return eIcicleError::INVALID_ARGUMENT;
      }
      size_t nof_queries;
      std::vector<std::vector<MerkleProof>> query_proofs;
      std::vector<F> final_poly;
      uint64_t pow_nonce;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&nof_queries, buffer_length, buffer, sizeof(size_t)));
      query_proofs.resize(nof_queries);
      for (size_t i = 0; i < nof_queries; ++i) {
        size_t nof_fri_rounds;
        ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&nof_fri_rounds, buffer_length, buffer, sizeof(size_t)));
        query_proofs[i].resize(nof_fri_rounds);
        for (size_t j = 0; j < nof_fri_rounds; ++j) {
          ICICLE_CHECK_IF_RETURN(BinarySerializeImpl<MerkleProof>::unpack_and_advance(buffer, buffer_length, query_proofs[i][j]));
        }
      }

      size_t final_poly_size;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&final_poly_size, buffer_length, buffer, sizeof(size_t)));
      final_poly.resize(final_poly_size);
      ICICLE_CHECK_IF_RETURN(
        memcpy_shift_source(final_poly.data(), buffer_length, buffer, final_poly_size * sizeof(F)));

      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&pow_nonce, buffer_length, buffer, sizeof(uint64_t)));
      FriProof<F> proof = FriProof<F>(std::move(query_proofs), std::move(final_poly), pow_nonce);
      obj = std::move(proof);
      return eIcicleError::SUCCESS;
    }
  };
} // namespace icicle
