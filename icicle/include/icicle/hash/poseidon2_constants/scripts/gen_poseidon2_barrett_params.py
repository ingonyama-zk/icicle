import os
import subprocess
import sys

possible_hash_width = [2, 3, 4, 8, 12, 16, 20, 24]
small_field_max_width = 64

fields = [
  # <field_name>, <is_field> <hash_width>, <prime>
  ["babybear",   [2, 3, 4, 8, 12, 16, 20, 24], 0x78000001],    # 15 * 2^27 + 1
  ["m31",        [2, 3, 4, 8, 12, 16, 20, 24], 0x7fffffff],   # 2^31 - 1
  # ["goldilocks", [2, 3, 4, 8, 12, 16, 20, 24], 0xffffffff00000001],    # 2^64 - 2^32 + 1
  ["stark252",   [2, 3, 4, 8], 0x800000000000011000000000000000000000000000000000000000000000001],    # 2^251 + 17 * 2^192 + 1
  ["bn254",      [2, 3, 4, 8], 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001],
  ["bls12_377",  [2, 3, 4, 8], 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001],
  ["bls12_381",  [2, 3, 4, 8], 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001],
  ["grumpkin",    [2, 3, 4, 8], 0x30644E72E131A029B85045B68181585D97816A916871CA8D3C208C16D87CFD47],
  ["bw6_761",    [2, 3, 4, 8], 0x1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001]
]

script_dir = os.path.dirname(os.path.abspath(__file__))
constants_files_dir = script_dir + '/../constants'
os.chdir(script_dir)  # Run in scripts directory because *.sage.py files are generated there.

for field in fields:
  poseidon2_h_file = f"{field[0]}_poseidon2.h"
  FILE_cpp = open(poseidon2_h_file, 'w')
  FILE_cpp.write(f'#pragma once\n')
  FILE_cpp.write(f'#ifndef {field[0].upper()}_POSEIDON2_H\n')
  FILE_cpp.write(f'#define {field[0].upper()}_POSEIDON2_H\n\n')
  FILE_cpp.write(f'#include <string>\n\n')
  FILE_cpp.write(f'namespace poseidon2_constants_{field[0]} {{\n\n')
  FILE_cpp.write(f'  /**\n')
  FILE_cpp.write(f'   * This inner namespace contains constants for running Poseidon2.\n')
  FILE_cpp.write(f'   * The number in the name corresponds to the arity of hash function\n')
  FILE_cpp.write(f'   */\n\n')
  for hash_width in possible_hash_width:
    field_width = len(bin(field[2]))-2
    gen_constants = True
    if field_width > small_field_max_width:  # large field
      if hash_width > max(field[1]):   # generate file with empty var's and arrays.
        gen_constants = False
    alpha = 0
    if not gen_constants:   # generate empty file to be concat later
      FILE_cpp_tmp = open(f'{field[0]}_poseidon2_{hash_width}_{alpha}.h', 'w')    # alpha = 0 just for later concatenation
      FILE_cpp_tmp.write(f'int full_rounds_{hash_width} =         0;\n')
      FILE_cpp_tmp.write(f'int half_full_rounds_{hash_width} =    0;\n')
      FILE_cpp_tmp.write(f'int partial_rounds_{hash_width} =      0;\n')
      FILE_cpp_tmp.write(f'int alpha_{hash_width} =               0;\n')
      FILE_cpp_tmp.write(f'static const std::string rounds_constants_{hash_width}[] = {{}};\n')
      FILE_cpp_tmp.write(f'static const std::string mds_matrix_{hash_width}[]      = {{}};\n')
      FILE_cpp_tmp.write(f'static const std::string partial_matrix_diagonal_{hash_width}[]  = {{}};\n\n')
      FILE_cpp_tmp.write(f'static const std::string partial_matrix_diagonal_m1_{hash_width}[]  = {{}};\n\n')
      FILE_cpp_tmp.close()
    else:
      gen_constants_cmd = f'sage poseidon2_barrett_params.sage {field[0]} {hash_width} {hex(field[2])}'
      print(f'command: {gen_constants_cmd}')
      result = subprocess.run([gen_constants_cmd], shell=True, capture_output=True, text=True)
      output_lines = result.stdout.strip().split('\n')
      for line in output_lines:
        print(line)
        if line.startswith("RESULT:"):
          full_round = int(line.split()[1])   # Not needed in this case.
          partial_round = int(line.split()[2])    # Not needed in this case.
          alpha = int(line.split()[3])
          break
    # exit()   # For DEBUG - run a single width of a single field

    with open(f'{field[0]}_poseidon2_{hash_width}_{alpha}.h', 'r') as f_params:
      FILE_cpp.write(f_params.read())
    FILE_cpp.write(f'\n')
  FILE_cpp.write(f'}}   // namespace poseidon2_constants_{field[0]} {{\n')
  FILE_cpp.write(f'#endif\n')
  FILE_cpp.close()
  print(f'rm -rf {constants_files_dir}/{poseidon2_h_file}')
  print(f'mv {poseidon2_h_file} {constants_files_dir}')
  os.system(f'rm -rf {constants_files_dir}/{poseidon2_h_file}')
  os.system(f'mv {poseidon2_h_file} {constants_files_dir}')
  os.system(f'rm -rf *.h')
  os.system(f'rm -rf *sage.py')
  os.system(f'rm -rf *_poseidon_rc_and_mds_matrix_*')
  # exit()   # For DEBUG - run a single field

exit()
  
