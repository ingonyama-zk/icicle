# Serialization

## Interface Description

The `BinarySerializer` interface provides methods for serializing and deserializing objects. It is a template class that can be used with various types. The primary methods include:

- `static eIcicleError serialized_size(const T& obj, size_t& size)`: Calculates the size required to serialize the object `obj` and stores it in `size`.

- `static eIcicleError serialize(std::byte* buffer, size_t buffer_length, const T& obj)`: Serializes the object `obj` into the provided `buffer` of length `buffer_length`.

- `static eIcicleError deserialize(const std::byte* buffer, size_t buffer_length, T& obj)`: Deserializes the data from the `buffer` into the object `obj`.

These methods return an `eIcicleError` indicating success or failure of the operation. Proper error handling should be implemented to ensure robustness.

## Example Usage

Here is an example of how to use the `BinarySerializer` for serialization and deserialization:

```cpp
#include "icicle/serialization.h"
#include "sumcheck/sumcheck_proof.h"

// Assume sumcheck_proof is an instance of SumcheckProof<scalar_t>
SumcheckProof<scalar_t> sumcheck_proof;

// Calculate serialized size
size_t proof_size = 0;
ICICLE_CHECK(BinarySerializer<SumcheckProof<scalar_t>>::serialized_size(sumcheck_proof, proof_size));

// Serialize the proof
std::vector<std::byte> proof_bytes(proof_size);
ICICLE_CHECK(BinarySerializer<SumcheckProof<scalar_t>>::serialize(proof_bytes.data(), proof_bytes.size(), sumcheck_proof));

// Deserialize the proof
SumcheckProof<scalar_t> deserialized_proof;
ICICLE_CHECK(BinarySerializer<SumcheckProof<scalar_t>>::deserialize(proof_bytes.data(), proof_bytes.size(), deserialized_proof));

```

This example demonstrates calculating the serialized size, performing serialization, and then deserialization.
