import sys
from hashlib import sha256

# --- Skyscraper round constant logic (from skyscraper.sage) ---
def SHA256(x):
    return sha256(x).digest()

def get_rcons_int(R, n, endianess='big'):
    sha_input = lambda i: int(i).to_bytes(4, endianess) + b'Skyscraper' + int(0).to_bytes(28-len("Skyscraper"), endianess)
    rcgen = lambda i: int.from_bytes(SHA256(sha_input(i)), endianess)
    return [rcgen(r) for r in range((R-2)*n)]

# Minimal BAR class for IL2FFE logic
class Bar:
    def __init__(self, p, n=1):
        self.p = p
        self.n = n
        if n == 1:
            self.F = GF(p)
        else:
            P.<x> = PolynomialRing(GF(p))
            modulus = P.irreducible_element(n)
            self.F = GF(p**n, name='a', modulus=modulus, repr='poly')
    def IL2FFE(self, Z):
        if self.n == 1:
            return self.F(Z[0])
        else:
            return self.F(Z)

def get_rcons(F, n, R, bar):
    rcons_int = get_rcons_int(R, n)
    rcons_field = [bar.IL2FFE(rcons_int[i:i + n]) for i in range(0, len(rcons_int), n)]
    return [F.zero()] + rcons_field + [F.zero()]

def to_hex(val, p):
    # Output as hex string, zero-padded to modulus size
    hexlen = (p.bit_length() + 3) // 4
    return f"0x{int(val):0{hexlen}x}"

def main():
    if len(sys.argv) < 5:
        print("Usage: sage skyscraper_params.sage <field_name> <modulus_hex> <R> <n1,n2,...>")
        exit(1)
    field_name = sys.argv[1]
    p = int(sys.argv[2], 16)
    R = int(sys.argv[3])
    n_list = [int(x) for x in sys.argv[4].split(',')]

    filename = f"../constants/{field_name}_skyscraper.h"
    with open(filename, "w") as f:
        f.write(f"#pragma once\n")
        f.write(f"#ifndef {field_name.upper()}_SKYSCRAPER_H\n")
        f.write(f"#define {field_name.upper()}_SKYSCRAPER_H\n\n")
        f.write(f"#include <string>\n\n")
        f.write(f"namespace skyscraper_constants_{field_name} {{\n\n")
        for n in n_list:
            bar = Bar(p, n)
            F = bar.F
            rcons = get_rcons(F, n, R, bar)
            f.write(f"static const std::string round_constants_{n}[] = {{\n")
            for elem in rcons:
                if n == 1:
                    f.write(f'  "{to_hex(elem, p)}",\n')
                else:
                    coeffs = elem.polynomial().list()
                    coeffs += [0] * (n - len(coeffs))
                    for c in coeffs:
                        f.write(f'  "{to_hex(c, p)}",\n')
            f.write("};\n\n")
        f.write("} // namespace\n")
        f.write(f"#endif // {field_name.upper()}_SKYSCRAPER_H\n")

if __name__ == "__main__":
    main() 