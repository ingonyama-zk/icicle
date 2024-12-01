#pragma once
#include <random>

inline std::mt19937 rand_generator = std::mt19937{std::random_device{}()};
static void seed_rand_generator(unsigned seed) { rand_generator.seed(seed); }