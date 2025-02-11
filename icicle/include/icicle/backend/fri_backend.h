#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/fri/fri_config.h"
#include "icicle/fri/fri_proof.h"
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/fields/field_config.h"
#include "icicle/backend/merkle/merkle_tree_backend.h"

using namespace field_config;

namespace icicle {

/**
 * @brief Abstract base class for FRI backend implementations.
 * @tparam F Field type used in the FRI protocol.
 */
template <typename F>
class FriBackend
{
public:
    /**
     * @brief Constructor that accepts an existing array of Merkle trees.
     *
     * @param folding_factor   The factor by which the codeword is folded each round.
     * @param stopping_degree  Stopping degree threshold for the final polynomial.
     */
    FriBackend(const size_t folding_factor,
               const size_t stopping_degree)
      : m_folding_factor(folding_factor)
      , m_stopping_degree(stopping_degree)
    {}

    virtual ~FriBackend() = default;

    /**
     * @brief Generate the FRI proof from given inputs.
     * 
     * @param fri_config            Configuration for FRI operations (e.g., proof-of-work bits, queries).
     * @param fri_transcript_config Configuration for encoding/hashing FRI messages (Fiat-Shamir).
     * @param input_data          Evaluations of the polynomial (or other relevant data).
     * @param fri_proof           (OUT) A FriProof object to store the proof's Merkle layers, final poly, etc.
     * @return eIcicleError       Error code indicating success or failure.
     */
    virtual eIcicleError get_fri_proof(
      const FriConfig& fri_config,
      const FriTranscriptConfig<F>& fri_transcript_config,
      const F* input_data,
      FriProof<F>& fri_proof
    ) = 0;

protected:
    const size_t m_folding_factor;
    const size_t m_stopping_degree;
};

/*************************** Backend Factory Registration ***************************/

/**
 * @brief A function signature for creating a FriBackend instance for a specific device.
 */
template <typename F>
using FriFactoryImpl =
    std::function<eIcicleError(const Device& device,const size_t folding_factor, const size_t stopping_degree, std::vector<MerkleTree>&& merkle_trees, std::shared_ptr<FriBackend<F>>& backend /*OUT*/)>;

/**
 * @brief Register a FRI backend factory for a specific device type.
 *
 * @param deviceType String identifier for the device type.
 * @param impl        A factory function that creates a FriBackend<F>.
 */
void register_fri_factory(const std::string& deviceType, FriFactoryImpl<scalar_t> impl);

/**
 * @brief Macro to register a FRI backend factory.
 *
 * This macro registers a factory function for a specific backend by calling
 * `register_fri_factory` at runtime.
 *
 */
#define REGISTER_FRI_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                                \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_fri) = []() -> bool {                                                                      \
      register_fri_factory(DEVICE_TYPE, FUNC);                                                                         \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle
