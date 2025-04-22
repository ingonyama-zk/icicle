#pragma once

#include <cstddef>
#include <fstream>
#include <vector>
#include <cstring>
#include "sumcheck/sumcheck_proof.h"
#include "fri/fri_proof.h"
#include "merkle/merkle_proof.h"

namespace icicle {
  namespace {
    inline eIcicleError
    memcpy_shift_destination(std::byte*& destination, size_t& remaining_length, const void* source, size_t copy_length)
    {
      if (remaining_length < copy_length) {
        ICICLE_LOG_ERROR << "memcpy_shift_destination failed: remaining_length < copy_length: " << remaining_length
                         << " < " << copy_length;
        return eIcicleError::INVALID_ARGUMENT;
      }
      std::memcpy(destination, source, copy_length);
      destination += copy_length;
      remaining_length -= copy_length;
      return eIcicleError::SUCCESS;
    }

    inline eIcicleError
    memcpy_shift_source(void* destination, size_t& remaining_length, std::byte*& source, size_t copy_length)
    {
      if (remaining_length < copy_length) {
        ICICLE_LOG_ERROR << "memcpy_shift_source failed: remaining_length < copy_length: " << remaining_length << " < "
                         << copy_length;
        return eIcicleError::INVALID_ARGUMENT;
      }
      std::memcpy(destination, source, copy_length);
      source += copy_length;
      remaining_length -= copy_length;
      return eIcicleError::SUCCESS;
    }
  }

  template <typename T>
  struct BinarySerializer {
    static eIcicleError serialized_size(const T& obj, size_t& size);
    static eIcicleError pack_and_advance(std::byte*& buffer, size_t& buffer_length, const T& obj);
    static eIcicleError unpack_and_advance(std::byte*& buffer, size_t& buffer_length, T* obj);
  };

  template <typename T>
  struct BinarySerializerBase {
    static eIcicleError serialize(std::byte* buffer, size_t buffer_length, const T& obj) {
      size_t size;
      ICICLE_CHECK_IF_RETURN(BinarySerializer<T>::serialized_size(obj, size));
      if (buffer_length != size) {
        ICICLE_LOG_ERROR << "Serialization failed: buffer_length != size: " << buffer_length << " != " << size;
        return eIcicleError::INVALID_ARGUMENT;
      }
      return BinarySerializer<T>::pack_and_advance(buffer, buffer_length, obj);
    }

    static eIcicleError deserialize(std::byte* buffer, size_t buffer_length, T* obj) {
      return BinarySerializer<T>::unpack_and_advance(buffer, buffer_length, obj);
    }

    static eIcicleError serialize_to_file(const std::string& path, const T& obj)
    {
      std::ofstream file(path, std::ios::binary);
      if (!file.is_open()) { return eIcicleError::INVALID_ARGUMENT; }

      size_t buffer_length;
      ICICLE_CHECK_IF_RETURN(BinarySerializer<T>::serialized_size(obj, buffer_length));
      std::vector<std::byte> buffer(buffer_length);
      std::byte* ptr = buffer.data();
      ICICLE_CHECK_IF_RETURN(BinarySerializer<T>::serialize(ptr, buffer_length, obj));
      file.write(reinterpret_cast<const char*>(buffer.data()), buffer_length);
      return eIcicleError::SUCCESS;
    }

    static eIcicleError deserialize_from_file(const std::string& path, T* obj)
    {
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file.is_open()) { return eIcicleError::INVALID_ARGUMENT; }

      std::streamsize size = file.tellg();
      if (size <= 0) { return eIcicleError::INVALID_ARGUMENT; }
      file.seekg(0, std::ios::beg);

      std::vector<std::byte> buffer(static_cast<size_t>(size));
      std::byte* buffer_ptr = buffer.data();
      if (!file.read(reinterpret_cast<char*>(buffer_ptr), size)) { return eIcicleError::INVALID_ARGUMENT; }

      return BinarySerializer<T>::deserialize(buffer_ptr, static_cast<size_t>(size), obj);
    }
  };

  template <typename S>
  struct BinarySerializer<SumcheckProof<S>> : BinarySerializerBase<SumcheckProof<S>> {
    static eIcicleError serialized_size(const SumcheckProof<S>& obj, size_t& size) {
      size = obj.get_nof_round_polynomials(); // nof_round_polynomials
      
      for (size_t i = 0; i < obj.get_nof_round_polynomials(); i++) {
        const auto& round_poly = obj.get_const_round_polynomial(i);
        size += sizeof(size_t); // nested vector length
        size += round_poly.size() * sizeof(S);
      }
      return eIcicleError::SUCCESS;
    }
    static eIcicleError pack_and_advance(std::byte*& buffer, size_t& buffer_length, const SumcheckProof<S>& obj) {
      size_t nof_round_polynomials = obj.get_nof_round_polynomials();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &nof_round_polynomials, sizeof(size_t)));
      for (size_t i = 0; i < nof_round_polynomials; i++) {
        const auto& round_poly = obj.get_const_round_polynomial(i);
        size_t round_poly_size = round_poly.size();
        ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &round_poly_size, sizeof(size_t)));
        ICICLE_CHECK_IF_RETURN(
          memcpy_shift_destination(buffer, buffer_length, round_poly.data(), round_poly_size * sizeof(S)));
      }
      return eIcicleError::SUCCESS;
    }
    static eIcicleError unpack_and_advance(std::byte*& buffer, size_t& buffer_length, SumcheckProof<S>* obj) {
      size_t nof_round_polynomials;
      std::vector<std::vector<S>> round_polynomials;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&nof_round_polynomials, buffer_length, buffer, sizeof(size_t)));

      round_polynomials.resize(nof_round_polynomials);
      for (size_t i = 0; i < nof_round_polynomials; ++i) {
        size_t round_poly_size;
        ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&round_poly_size, buffer_length, buffer, sizeof(size_t)));

        size_t byte_size = round_poly_size * sizeof(S);
        round_polynomials[i].resize(round_poly_size);
        ICICLE_CHECK_IF_RETURN(memcpy_shift_source(round_polynomials[i].data(), buffer_length, buffer, byte_size));
      }

      SumcheckProof<S> proof = SumcheckProof<S>(std::move(round_polynomials));
      *obj = std::move(proof);
      return eIcicleError::SUCCESS;
    }
  };

  template <>
  struct BinarySerializer<MerkleProof> : BinarySerializerBase<MerkleProof> {
    static eIcicleError serialized_size(const MerkleProof& obj, size_t& size) {
      size = sizeof(bool);                       // pruned
      size += sizeof(uint64_t);                  // leaf_index
      size += sizeof(size_t);                    // leaf_size
      auto [leaf, leaf_size, leaf_index] = obj.get_leaf();
      size += leaf_size * sizeof(std::byte); // leaf
      size += sizeof(size_t);   // root_size
      auto [root, root_size] = obj.get_root();                 
      size += root_size * sizeof(std::byte); // root
      size += sizeof(size_t);                    // path_size
      auto [path, path_size] = obj.get_path();
      size += path_size * sizeof(std::byte); // path
      return eIcicleError::SUCCESS;
    }
    static eIcicleError pack_and_advance(std::byte*& buffer, size_t& buffer_length, const MerkleProof& obj) {
      bool pruned = obj.is_pruned();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &pruned, sizeof(bool)));

      auto [leaf, leaf_size, leaf_index] = obj.get_leaf();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &leaf_index, sizeof(uint64_t)));

      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &leaf_size, sizeof(size_t)));
      ICICLE_CHECK_IF_RETURN(
        memcpy_shift_destination(buffer, buffer_length, leaf, leaf_size * sizeof(std::byte)));

      auto [root, root_size] = obj.get_root();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &root_size, sizeof(size_t)));
      ICICLE_CHECK_IF_RETURN(
        memcpy_shift_destination(buffer, buffer_length, root, root_size * sizeof(std::byte)));

      auto [path, path_size] = obj.get_path();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &path_size, sizeof(size_t)));
      ICICLE_CHECK_IF_RETURN(
        memcpy_shift_destination(buffer, buffer_length, path, path_size * sizeof(std::byte)));

      return eIcicleError::SUCCESS;
    }
    static eIcicleError unpack_and_advance(std::byte*& buffer, size_t& buffer_length, MerkleProof* obj) {
      bool pruned;
      int64_t leaf_idx;
      std::vector<std::byte> leaf;
      std::vector<std::byte> root;
      std::vector<std::byte> path;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&pruned, buffer_length, buffer, sizeof(bool)));

      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&leaf_idx, buffer_length, buffer, sizeof(uint64_t)));

      size_t leaf_size;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&leaf_size, buffer_length, buffer, sizeof(size_t)));
      leaf.resize(leaf_size);
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(leaf.data(), buffer_length, buffer, leaf_size * sizeof(std::byte)));

      size_t root_size;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&root_size, buffer_length, buffer, sizeof(size_t)));
      root.resize(root_size);
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(root.data(), buffer_length, buffer, root_size * sizeof(std::byte)));

      size_t path_size;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&path_size, buffer_length, buffer, sizeof(size_t)));
      path.resize(path_size);
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(path.data(), buffer_length, buffer, path_size * sizeof(std::byte)));

      MerkleProof proof = MerkleProof(pruned, leaf_idx, std::move(leaf), std::move(root), std::move(path));
      *obj = std::move(proof);

      return eIcicleError::SUCCESS;
    }
  };

  template <typename F>
  struct BinarySerializer<FriProof<F>> : BinarySerializerBase<FriProof<F>> {
    static eIcicleError serialized_size(const FriProof<F>& obj, size_t& size) {
      size = sizeof(size_t); // nof_queries
      size_t nof_queries = obj.get_nof_queries();
      size_t nof_rounds = obj.get_nof_rounds();
      for (size_t i = 0; i < nof_queries; i++) {
        size += sizeof(size_t); // nof_fri_rounds
        for (size_t j = 0; j < nof_rounds; j++) {
          const auto& proof = obj.get_query_proof_slot(i, j);
          size_t proof_size = 0;
          ICICLE_CHECK_IF_RETURN(BinarySerializer<MerkleProof>::serialized_size(proof, proof_size));
          size += proof_size;
        }
      }
      size += sizeof(size_t); // final_poly_size
      size += obj.get_final_poly_size() * sizeof(F);
      size += sizeof(uint64_t); // pow_nonce

      return eIcicleError::SUCCESS;
    }
    static eIcicleError pack_and_advance(std::byte*& buffer, size_t& buffer_length, const FriProof<F>& obj) {
      size_t query_proofs_size = obj.get_nof_queries();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &query_proofs_size, sizeof(size_t)));
      for (size_t i = 0; i < query_proofs_size; i++) {
        size_t nof_rounds = obj.get_nof_rounds();
        ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &nof_rounds, sizeof(size_t)));
        for (size_t j = 0; j < nof_rounds; j++) {
          const auto& proof = obj.get_query_proof_slot(i, j);
          ICICLE_CHECK_IF_RETURN(BinarySerializer<MerkleProof>::pack_and_advance(buffer, buffer_length, proof));
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
    static eIcicleError unpack_and_advance(std::byte*& buffer, size_t& buffer_length, FriProof<F>* obj) {
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
          auto proof = &query_proofs[i][j];
          ICICLE_CHECK_IF_RETURN(BinarySerializer<MerkleProof>::unpack_and_advance(buffer, buffer_length, proof));
        }
      }

      size_t final_poly_size;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&final_poly_size, buffer_length, buffer, sizeof(size_t)));
      final_poly.resize(final_poly_size);
      ICICLE_CHECK_IF_RETURN(
        memcpy_shift_source(final_poly.data(), buffer_length, buffer, final_poly_size * sizeof(F)));

      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&pow_nonce, buffer_length, buffer, sizeof(uint64_t)));
      FriProof<F> proof = FriProof<F>(std::move(query_proofs), std::move(final_poly), pow_nonce);
      *obj = std::move(proof);
      return eIcicleError::SUCCESS;
    }
  };
} // namespace icicle