from sympy import factorint
from sympy.ntheory.modular import crt
import math

# Prime fields - find generator

from sympy import factorint

def is_primitive_root(g, q, phi, factors):
    """Checks if g is a primitive root mod q by verifying its order."""
    for p in factors:
        if pow(g, phi // p, q) == 1:
            return False
    return True

def find_generator(prime):
    """Finds a primitive root (generator) of F_p*."""
    if prime <= 2:
        return 1

    phi = prime - 1  # The order of F_p*
    factors = list(factorint(phi).keys())  # Prime factorization of p-1

    for g in range(2, prime):
        if is_primitive_root(g, prime, phi, factors):
            return g  # Found a valid generator!
    return None

def largest_power_of_two_subgroup_in_prime_field(p):
    """Computes logn, the largest power of 2 that divides (p-1)."""
    factors = factorint(p - 1)  # Factorize p-1
    return factors.get(2, 0)  # Return the exponent of 2, or 0 if not present


def find_primitive_root_of_unity(prime, logn):
    
    """Finds a primitive 2^logn-th root of unity modulo prime."""
    generator = find_generator(prime)
    order = (prime - 1) // (2 ** logn)  # Compute exponent for the root of unity
    return pow(generator, order, prime)

# CRT - from RNS to Direct representation

from sympy.ntheory.modular import crt

def from_rns(p_list, g_list):
    """
    Computes a generator of Z_q^* given generators for Z_p^* using SymPy's CRT.

    Arguments:
    - p_list: List of primes [p1, p2, ..., pn].
    - g_list: Corresponding generators [g1, g2, ..., gn].

    Returns:
    - A single generator for Z_q^* using the Chinese Remainder Theorem (CRT).
    """
    assert len(p_list) == len(g_list), "Lists must be of the same length"

    # Compute the modulus q = p1 * p2 * ... * pn
    q = 1
    for p in p_list:
        q *= p

    # Compute CRT for all primes and generators
    g, _ = crt(p_list, g_list)

    return g % q  # Ensure g is within the correct range


def to_32b_limbs_str(value):
    """Prints a value in 32-bit limbs starting from LSB first."""
    # Compute the number of 32-bit limbs needed
    num_limbs = (value.bit_length() + 31) // 32  # Round up to the nearest limb count

    # Extract 32-bit limbs (starting from LSB first)
    limbs = [(value >> (32 * i)) & 0xFFFFFFFF for i in range(num_limbs)]

    # Format as a string
    return "{ " + ", ".join(f"0x{limb:08x}" for limb in limbs) + " }"


######################################################

def labrador():
    pbb = 0x78000001            # babybear
    pkb = 0x7f000001            # koalabear
    q = pbb*pkb
    print(f"Labrador: {hex(q)}")
    print(f"Labrador: {to_32b_limbs_str(q)}")
    print(f"Labrador bitcount={q.bit_length()}")

    bb_max_rou_order = largest_power_of_two_subgroup_in_prime_field(pbb)
    kb_max_rou_order = largest_power_of_two_subgroup_in_prime_field(pkb)
    print(f"Max order of 2 in koalabear: {kb_max_rou_order}")
    print(f"Max order of 2 in babybear: {bb_max_rou_order}")
    max_order_in_labrador = min(bb_max_rou_order, kb_max_rou_order)
    print(f"Max order of 2 in labrador: {max_order_in_labrador}")
    
    # bb_rou_27 = 0x00000089 # Baby bear
    # bb_rou = pow(bb_rou_27, 8, pbb) #w^8 mod p1 for w or order logn=27
    # kb_rou = 0x6ac49f88 # Koala bear
    
    bb_rou = find_primitive_root_of_unity(pbb, max_order_in_labrador)
    kb_rou = find_primitive_root_of_unity(pkb, max_order_in_labrador)
    print(f"babybear rou: {hex(bb_rou)}")
    print(f"koalabear rou: {hex(kb_rou)}")
    q_rou = from_rns([pbb, pkb], [bb_rou, kb_rou])
    
    print(f"(logn={max_order_in_labrador}) w^n mod q = {pow(q_rou, 1<<max_order_in_labrador, q)}")
    print(f"Rou in the ring q = {to_32b_limbs_str(q_rou)}")
    
    # precompute RNS Wi
    M0 = pkb # pbb*pkb / pbb
    M1 = pbb # pbb*pkb / pkb
    M0_inv = pow(M0, -1, pbb)
    M1_inv = pow(M1, -1, pkb)
    W0 = M0*M0_inv % q
    W1 = M1*M1_inv % q
    print(f"W0 = {to_32b_limbs_str(W0)}")
    print(f"W1 = {to_32b_limbs_str(W1)}")
    
    # Test conversion to RNS and back for a random number
    import random
    x = random.randint(0, q-1)
    x0 = x % pbb
    x1 = x % pkb
    print(f"x = {x}")
    print(f"x0 = {x0}")
    print(f"x1 = {x1}")
    x_ = (W0*x0 + W1*x1) % q
    print(f"x_ = {x_}")
    assert x == x_, "Wi computation failed"    

def greyhound():
    Pbb = 0x78000001
    Pkb = 0x7f000001
    Ptb = (2**32)-(2**30)+1
    Pcb = (2**30)+(2**25)+1
    Pgb = (2**29)-(2**26)+1
    q_greyhound = Pbb*Pkb*Ptb*Pcb*Pgb
    # q_greyhound = q_greyhound*q_greyhound # TODO remove
    print(f"Greyhound: {hex(q_greyhound)}")
    print(f"Greyhound: {to_32b_limbs_str(q_greyhound)}")
    print(f"Greyhound bitcount={q_greyhound.bit_length()}")

    # compute root of unity of order logn=24 for greyhound

######################################################

if __name__ == "__main__":
    labrador()
    # greyhound()