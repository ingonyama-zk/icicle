#!/usr/bin/env python3

import galois
import numpy as np

# pip install poseidon-hash
from poseidon import round_constants as rc, round_numbers as rn

t = 3
alpha = 5
# p = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001 # bls12-381
p = 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001 # bls12-377
prime_bit_len = 253
field_bytes = 32

# leave set to -1 if not sure
full_round = -1
half_full_round = full_round // 2
# leave set to -1 if not sure
partial_round = -1

security_level = 128

def flatten(xss):
    return [x for xs in xss for x in xs]

if __name__ == "__main__":
    if full_round == -1 or partial_round == -1:
        _, partial_round, half_full_round = rn.calc_round_numbers(prime_bit_len, security_level, t, alpha, True)
        print("Half full rounds:", half_full_round)
        print("Partial rounds:", partial_round)

    print("Loading galois... This might take several minutes")
    field_p = galois.GF(p)
    print("Galois loaded")
    mds_matrix = rc.mds_matrix_generator(field_p, t)
    non_opt_rc = rc.calc_round_constants(t, full_round, partial_round, p, field_p, alpha, prime_bit_len)
    split_rc = [field_p(x.tolist()) for x in np.array_split(non_opt_rc, len(non_opt_rc) / t)]
    opt_rc = rc.optimized_rc(split_rc, half_full_round, partial_round, mds_matrix)
    pre_matrix, sparse_matrices = rc.optimized_matrix(mds_matrix, partial_round, field_p)

    sparse_aligned = []
    for m in sparse_matrices:
        m = flatten(m.tolist())
        for j in range(0, t * t, t):
                sparse_aligned.append(m[j])
        for j in range(1, t):
                sparse_aligned.append(m[j])

    with open("constants.bin", "wb") as constants_file:
        for l in [opt_rc, flatten(mds_matrix), flatten(pre_matrix), sparse_aligned]:
            for c in l:
                constants_file.write(int(c).to_bytes(32, byteorder='little'))