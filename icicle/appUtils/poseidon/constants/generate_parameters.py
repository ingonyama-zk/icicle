#!/usr/bin/env python3

# 0.3.7
import galois
import numpy as np

# pip install poseidon-hash
from poseidon import round_constants as rc, round_numbers as rn

# Modify these
arity = 11
p = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001 # bls12-381
# p = 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001 # bls12-377
# p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001 # bn254
# p = 0x1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001 # bw6-761
prime_bit_len = 255
field_bytes = 32

# leave set to -1 if not sure
full_round = -1
half_full_round = full_round // 2
# leave set to -1 if not sure
partial_round = -1

security_level = 128

# May speed up Galois significantly. If not sure - set it to None
# You can get primitive element fast by using sage
# p = ...
# F = GF(p)
# F.primitive_element()
#
# primitive_element = None
primitive_element = 7 # bls12-381
# primitive_element = 22 # bls12-377
# primitive_element = 5 # bn254
# primitive_element = 15 # bw6-761

# currently we only support alpha 5, if you need alpha other than 5 - feal free to reach out
alpha = 5
t = arity + 1

def flatten(xss):
    return [x for xs in xss for x in xs]

if __name__ == "__main__":
    if full_round == -1 or partial_round == -1:
        full_round, partial_round, half_full_round = rn.calc_round_numbers(prime_bit_len, security_level, t, alpha, True)
        print("Half full rounds:", half_full_round)
        print("Partial rounds:", partial_round)

    print("Loading galois... This might take from several minutes to an hour")
    field_p = galois.GF(p, 1, verify=False, primitive_element=primitive_element)
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
                constants_file.write(int(c).to_bytes(field_bytes, byteorder='little'))