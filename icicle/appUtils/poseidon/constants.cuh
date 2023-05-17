#pragma once

#include <map>
#include <cassert>
#include <fstream>
#include <string>

#include "../../curves/curve_config.cuh"

const std::map<uint, uint> ARITY_TO_ROUND_NUMBERS = {
    {2, 55},
    {4, 56},
    {8, 57},
    {11, 57},
};

// TO-DO: change to mapping
const uint FULL_ROUNDS_DEFAULT = 4;

static void get_round_numbers(const uint arity, uint * partial_rounds, uint * half_full_rounds) {
    auto partial = ARITY_TO_ROUND_NUMBERS.find(arity);
    assert(partial != ARITY_TO_ROUND_NUMBERS.end());

    *partial_rounds = partial->second;
    *half_full_rounds = FULL_ROUNDS_DEFAULT;
}

scalar_t * load_scalars_from_file(const std::string filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    assert(file.is_open());

    size_t size = file.tellg();
    scalar_t * scalars = static_cast< scalar_t * >(malloc(size));
    file.read(reinterpret_cast<char*>(scalars), size);
    file.close();

    return scalars;
}

scalar_t * load_constants(const uint arity) {
    return load_scalars_from_file(std::string("constants/constants_") + std::to_string(arity));
}