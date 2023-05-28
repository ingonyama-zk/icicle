#pragma once

#include <map>
#include <stdexcept>
#include <cassert>

#include "constants/constants_2.h"
#include "constants/constants_4.h"
#include "constants/constants_8.h"
#include "constants/constants_11.h"

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

// TO-DO: for now, the constants are only generated in bls12_381
template <typename S>
S * load_constants(const uint arity) {
    unsigned char * constants;
    switch (arity) {
        case 2:
            constants = constants_2;
            break;
        case 4:
            constants = constants_4;
            break;
        case 8:
            constants = constants_8;
            break;
        case 11:
            constants = constants_11;
            break;
        default:
            throw std::invalid_argument( "unsupported arity" );
    }
    return reinterpret_cast< S * >(constants);
}