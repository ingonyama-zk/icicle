

#include <iostream>
#include <cstring>
#include <chrono>
#include <string>
#include "hash/blake2/blake2.h"
#include "hash/blake2/blake2-impl.h"
#include "hash/hash.h"
#include <chrono>

using namespace blake2s_hash;

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) \
  printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

// Ensure BLAKE2S_KEYBYTES, BLAKE2S_OUTBYTES, and blake2s function are defined appropriately

int main(void)
{
    using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
    using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
    
    // Set key to an empty string
    const char *key = "";  // Empty string as the key
    size_t keylen = 0;     // Length of the key is 0

    // The string to hash
    const char *test_string = "0123456789";
    size_t test_string_len = strlen(test_string);

    // Test simple API
    START_TIMER(blake_ref)
    {
        uint8_t hash[BLAKE2S_OUTBYTES];
        // Pass the empty key with keylen = 0
        blake2s(hash, BLAKE2S_OUTBYTES, test_string, test_string_len, key, keylen);

        // Expected hash value for "0123456789" with an empty key as a string
        const char *expected_hash_str = "410381eb72313f23f9f62478d62ec7635f4166ab5e53a20af5c9e8f7ee445de8";

        // Convert computed hash to a string
        char computed_hash_str[BLAKE2S_OUTBYTES * 2 + 1]; // Two characters per byte + null terminator
        for (size_t i = 0; i < BLAKE2S_OUTBYTES; i++) {
            sprintf(&computed_hash_str[i * 2], "%02x", hash[i]);
        }
        computed_hash_str[BLAKE2S_OUTBYTES * 2] = '\0'; // Null terminator

        // Print the computed and expected hash strings
        std::cout << "Computed hash: " << computed_hash_str << std::endl;
        std::cout << "Expected hash: " << expected_hash_str << std::endl;

        if (strcmp(computed_hash_str, expected_hash_str) != 0) {
            std::cerr << "Hash mismatch!" << std::endl;
            goto fail;
        }
    }
    END_TIMER(blake_ref, "Blake Ref Timer")

    puts("ok");
    return 0;
fail:
    puts("error");
    return -1;
}
