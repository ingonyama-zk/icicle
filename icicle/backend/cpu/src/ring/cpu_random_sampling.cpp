#include "icicle/backend/vec_ops_backend.h"
#include <random>
#include <cstring>

eIcicleError cpu_random_sampling(const Device& device, uint64_t size, bool fast_mode, const std::byte* seed, uint64_t seed_len, const VecOpsConfig& config, field_t* output)
{
    std::seed_seq seed_seq(
        reinterpret_cast<const uint8_t*>(seed),
        reinterpret_cast<const uint8_t*>(seed) + seed_len
    );
    std::mt19937_64 rng(seed_seq);

    for (uint64_t i = 0; i < size; ++i) {
        output[i] = field_t::random(rng);
    }

    return eIcicleError::SUCCESS;
}

eIcicleError cpu_random_sampling_rq(const Device& device, uint64_t size, bool fast_mode, const std::byte* seed, uint64_t seed_len, const VecOpsConfig& config, Rq* output)
{
    return eIcicleError::SUCCESS;
}

REGISTER_RING_ZQ_RANDOM_SAMPLING_BACKEND("CPU", cpu_random_sampling);
REGISTER_RING_RQ_RANDOM_SAMPLING_BACKEND("CPU", cpu_random_sampling_rq);