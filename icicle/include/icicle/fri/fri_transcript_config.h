#pragma once

#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include <cstring>      // for std::strlen
#include <vector>
#include <algorithm>

namespace icicle {

/**
 * @brief Configuration for encoding and hashing messages in the FRI protocol.
 *
 * @tparam F Type of the field element (e.g., prime field or extension field elements).
 */
template <typename F>
class FriTranscriptConfig
{
public:
    // Default Constructor
    FriTranscriptConfig()
      : m_hasher(create_keccak_256_hash()),
        m_domain_separator_label({}),
        m_commit_phase_label({}),
        m_nonce_label(cstr_to_bytes("nonce")),
        m_public({}),
        m_seed_rng(F::zero())
    {
    }

    // Constructor with std::byte vectors for labels
    FriTranscriptConfig(
        Hash hasher,
        std::vector<std::byte>&& domain_separator_label,
        std::vector<std::byte>&& round_challenge_label,
        std::vector<std::byte>&& commit_phase_label,
        std::vector<std::byte>&& nonce_label,
        std::vector<std::byte>&& public_state,
        F seed_rng)
      : m_hasher(std::move(hasher)),
        m_domain_separator_label(std::move(domain_separator_label)),
        m_round_challenge_label(std::move(round_challenge_label)),
        m_commit_phase_label(std::move(commit_phase_label)),
        m_nonce_label(std::move(nonce_label)),
        m_public(std::move(public_state)),
        m_seed_rng(seed_rng)
    {
    }

    // Constructor with const char* arguments
    FriTranscriptConfig(
        Hash hasher,
        const char* domain_separator_label,
        const char* round_challenge_label,
        const char* commit_phase_label,
        const char* nonce_label,
        std::vector<std::byte>&& public_state,
        F seed_rng)
      : m_hasher(std::move(hasher)),
        m_domain_separator_label(cstr_to_bytes(domain_separator_label)),
        m_round_challenge_label(cstr_to_bytes(round_challenge_label)),
        m_commit_phase_label(cstr_to_bytes(commit_phase_label)),
        m_nonce_label(cstr_to_bytes(nonce_label)),
        m_public(std::move(public_state)),
        m_seed_rng(seed_rng)
    {
    }

    // Move Constructor
    FriTranscriptConfig(FriTranscriptConfig&& other) noexcept
      : m_hasher(std::move(other.m_hasher)),
        m_domain_separator_label(std::move(other.m_domain_separator_label)),
        m_round_challenge_label(std::move(other.m_round_challenge_label)),
        m_commit_phase_label(std::move(other.m_commit_phase_label)),
        m_nonce_label(std::move(other.m_nonce_label)),
        m_public(std::move(other.m_public)),
        m_seed_rng(other.m_seed_rng)
    {
    }

    const Hash& get_hasher() const { return m_hasher; }

    const std::vector<std::byte>& get_domain_separator_label() const {
        return m_domain_separator_label;
    }

    const std::vector<std::byte>& get_round_challenge_label() const {
        return m_round_challenge_label;
    }

    const std::vector<std::byte>& get_commit_phase_label() const {
        return m_commit_phase_label;
    }

    const std::vector<std::byte>& get_nonce_label() const {
        return m_nonce_label;
    }

    const std::vector<std::byte>& get_public_state() const {
        return m_public;
    }

    const F& get_seed_rng() const { return m_seed_rng; }

private:

    // Hash function used for randomness generation.
    Hash m_hasher;

    // Common transcript labels
    std::vector<std::byte> m_domain_separator_label;
    std::vector<std::byte> m_round_challenge_label;

    // FRI-specific labels
    std::vector<std::byte> m_commit_phase_label;
    std::vector<std::byte> m_nonce_label;
    std::vector<std::byte> m_public;

    // Seed for initializing the RNG.
    F m_seed_rng;


    static inline std::vector<std::byte> cstr_to_bytes(const char* str)
    {
        if (str == nullptr) return {};
        const size_t length = std::strlen(str);
        return {
            reinterpret_cast<const std::byte*>(str),
            reinterpret_cast<const std::byte*>(str) + length
        };
    }
};

} // namespace icicle
