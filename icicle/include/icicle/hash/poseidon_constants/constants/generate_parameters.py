#!/usr/bin/env python3

# 0.3.7
import galois
import numpy as np

import os

# pip install poseidon-hash
from poseidon import round_constants as rc, round_numbers as rn
import poseidon
print(f'poseidon.__file__ = {poseidon.__file__}')

# Modify these
arities = [3, 5, 9, 12]
#=======
# field = "m31"
# p = 2 ** 31 - 1
# prime_bit_len = 32
# field_bytes = 4
#=======
# field = "babybear"
# p = 2 ** 31 - 2 ** 27 + 1
# prime_bit_len = 32
# field_bytes = 4
#=======
field = "koalabear"
p = 2**31 - 2**24 + 1
prime_bit_len = 32
field_bytes = 4
#=======
# field = "bn254"
# p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001 # bn254
# prime_bit_len = 254
# field_bytes = 32
#=======
# field = "stark252"
# p = 2 ** 251 + 17 * (2 **192) + 1 # stark252
# prime_bit_len = 252
# field_bytes = 32
#=======
# field = "bls12_377"
# p = 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001
# prime_bit_len = 377
# field_bytes = 48
#=======
# field = "bls12_381"
# p = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001
# prime_bit_len = 381
# field_bytes = 48
#=======
# field = "bw6_761"
# p = 0x1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001
# prime_bit_len = 761
# field_bytes = 96
#=======
# p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47 # grumpkin

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
primitive_element = None
# primitive_element = 7 # bls12-381
# primitive_element = 22 # bls12-377
# primitive_element = 5 # bn254
# primitive_element = 15 # bw6-761
# primitive_element = 3 # grumpkin

# currently we only support alpha 5, if you need alpha other than 5 - feal free to reach out
alpha = 5

def flatten(xss):
    return [x for xs in xss for x in xs]

if __name__ == "__main__":
    rounds_numbers_txt_file = "rounds_numbers_" + field + ".txt"
    os.system('echo "" > ' + rounds_numbers_txt_file)
    for arity in arities:
      if full_round == -1 or partial_round == -1:
        full_round, partial_round, half_full_round = rn.calc_round_numbers(prime_bit_len, security_level, arity, alpha, True)
        os.system('arity='+str(arity)+'; nof_full_rounds='+str(full_round)+';  rounds_numbers_txt_file='+rounds_numbers_txt_file+'; echo "int full_rounds_$arity        $nof_full_rounds" >> $rounds_numbers_txt_file')
        os.system('arity='+str(arity)+'; nof_half_full_rounds='+str(half_full_round)+';  rounds_numbers_txt_file='+rounds_numbers_txt_file+'; echo "int half_full_rounds_$arity        $nof_half_full_rounds" >> $rounds_numbers_txt_file')
        os.system('arity='+str(arity)+'; nof_partial_rounds='+str(partial_round)+';  rounds_numbers_txt_file='+rounds_numbers_txt_file+'; echo "int partial_rounds_$arity        $nof_partial_rounds" >> $rounds_numbers_txt_file')
        os.system('echo "" >> ' + rounds_numbers_txt_file)
        print("Half full rounds:", half_full_round)
        print("Partial rounds:", partial_round)
        if arity != arities[-1]:
          full_round = -1
          partial_round = -1

    print("Loading galois... This might take from several minutes to an hour")
    field_p = galois.GF(p, 1, verify=False, primitive_element=primitive_element)
    print("Galois loaded")
    for arity in arities:
      mds_matrix = rc.mds_matrix_generator(field_p, arity)
      print("mds_matrix done")
      non_opt_rc = rc.calc_round_constants(arity, full_round, partial_round, p, field_p, alpha, prime_bit_len)
      split_rc = [field_p(x.tolist()) for x in np.array_split(non_opt_rc, len(non_opt_rc) / arity)]
      opt_rc = rc.optimized_rc(split_rc, half_full_round, partial_round, mds_matrix)
      print("round constants done")
      pre_matrix, sparse_matrices = rc.optimized_matrix(mds_matrix, partial_round, field_p)
      print("pre_matrix and sparse_matrix done")
      # print(f'Print pre- and sparse- matrices')
      # print(f'===============================')
      # print(f'len(pre_matrix) = {len(pre_matrix)}')
      # print(f'len(sparse_matrices) = {len(sparse_matrices)}')
      # print(f'pre_matrix = {pre_matrix}')
      # print(f'sparse_matrices = {sparse_matrices}')
      # print(f'opt_rc = {opt_rc}')
  
      sparse_aligned = []
      for m in sparse_matrices:
          m = flatten(m.tolist())
          for j in range(0, arity * arity, arity):
              sparse_aligned.append(m[j])
          for j in range(1, arity):
              sparse_aligned.append(m[j])
      
      constants_bin_file = "constants_" + str(arity) + "_" + field + ".bin"
      constants_txt_file = "constants_" + str(arity) + "_" + field + ".txt"
      with open(constants_bin_file, "wb") as constants_file:
          for l in [opt_rc, flatten(mds_matrix), flatten(pre_matrix), sparse_aligned]:
              for c in l:
                  constants_file.write(int(c).to_bytes(field_bytes, byteorder='little'))
      # Generate text file from a binary one.
      # Then use manual editing on the text file to generate <field>_poseidon.h
      exit_status = os.system('xxd -i -c 12 ' + constants_bin_file + " > " + constants_txt_file)
      print(exit_status)


