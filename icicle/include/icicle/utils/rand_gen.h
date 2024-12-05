#pragma once
#include <random>

inline std::mt19937 rand_generator = std::mt19937{std::random_device{}()};
static void seed_rand_generator(unsigned seed) { rand_generator.seed(seed); }

/**
 * @brief Generate random unsigned integer in range (inclusive)
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Random (uniform distribution) unsigned integer s.t. min <= integer <= max.
 */
static uint32_t rand_uint_32b(uint32_t min = 0, uint32_t max = UINT32_MAX)
{
  std::uniform_int_distribution<uint32_t> dist(min, max);
  return dist(rand_generator);
}