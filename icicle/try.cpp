#include <iostream>
#include <cstdlib>

#include "/home/administrator/users/danny/github/icicle/danny_poseidon2_v3.2/icicle/icicle/include/icicle/hash/poseidon2_constants/constants/bn254_poseidon2.h"

int main() {
    const char* hexString = "ab123456";

    unsigned long hexNumber = strtoul(hexString, nullptr, 16);

    std::cout << "Hex number: " << hexNumber << std::endl;
    std::cout << "Hex number: " << std::hex << hexNumber << std::endl;

    return 0;
}

