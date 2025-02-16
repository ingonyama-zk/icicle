from sympy import factorint
from sympy.ntheory.modular import crt
import math

# Prime fields - find generator

def is_generator(g, p, factors):
    """Checks if g is a primitive root modulo p by verifying its order."""
    for q in factors:
        if pow(g, (p - 1) // q, p) == 1:  # g^( (p-1)/q ) should NOT be 1
            return False
    return True

def find_generator(p):
    """Finds a primitive root (generator) of the multiplicative group F_p*."""
    if p <= 2:
        return 1  # The only generator of F_2* is 1
    
    phi = p - 1  # The order of F_p*
    factors = list(factorint(phi).keys())  # Prime factorization of p-1

    for g in range(2,p):
        if is_generator(g, p, factors):
            return g  # Found a valid generator!
    return None

# CRT - from RNS to Direct representation

def from_rns(p1, p2, g1, g2):
    """Computes a generator of Z_q^* given generators for Z_p1^* and Z_p2^* using SymPy's CRT."""
    q = p1 * p2
    
    # Compute CRT
    g, _ = crt([p1, p2], [g1, g2])
    
    return g % q  # Ensure g is within the correct range

def is_primitive_root(g, q, phi, factors):
    """Checks if g is a primitive root mod q by verifying its order."""
    for p in factors:
        if pow(g, phi // p, q) == 1:  # If g^((p1-1)(p2-1)/p) ≡ 1 (mod q), g is NOT a generator
            return False
    return True

def find_roots_of_unity(p1, p2, generator):
    """Finds the n-th primitive root of unity modulo q and all roots."""
    q = p1 * p2
    factors_p1 = factorint(p1-1)
    factors_p2 = factorint(p2-1) 
    print(f"factors-p1={factors_p1}")
    print(f"factors-p2={factors_p2}")
    
     # Merge dictionaries, taking the max exponent for each prime
    combined_factors = {}
    for prime in set(factors_p1.keys()).union(factors_p2.keys()):
        combined_factors[prime] = max(factors_p1.get(prime, 0), factors_p2.get(prime, 0))
        
    print(f"combined_factors={combined_factors}")

    # Compute the final product
    multiplicative_group_size = 1
    for prime, exponent in combined_factors.items():
        multiplicative_group_size *= prime ** exponent

    max_power_of_two = 2 ** combined_factors.get(2, 0)

    # Find a primitive root of q
    if not is_primitive_root(generator, q, multiplicative_group_size, combined_factors):
        raise ValueError("Invalid generator")

    # Compute the primitive n-th root of unity
    omega = pow(generator, multiplicative_group_size // max_power_of_two, q)

    return max_power_of_two, omega, q

def to_32b_limbs_str(value):
    """Prints a value in 32-bit limbs starting from LSB first."""
    # Compute the number of 32-bit limbs needed
    num_limbs = (value.bit_length() + 31) // 32  # Round up to the nearest limb count

    # Extract 32-bit limbs (starting from LSB first)
    limbs = [(value >> (32 * i)) & 0xFFFFFFFF for i in range(num_limbs)]

    # Format as a string
    return "{ " + ", ".join(f"0x{limb:08x}" for limb in limbs) + " }"

# Example usage:
p1 = 0x78000001
p2 = 0x7f000001
g1 = find_generator(p1)  # Example generator of Z_p1^*
g2 = find_generator(p2)  # Example generator of Z_p2^*
print(f"Generator of p1: {hex(g1)}")
print(f"Generator of p2: {hex(g2)}")

g = from_rns(p1, p2, g1, g2)
print(f"Generator of a multiplicative subgroup (<Zq*) of Z_q: {g}")

n, omega, q = find_roots_of_unity(p1, p2, g)
omega_limbs = to_32b_limbs_str(omega)

# Print results
print(f"logn (power-of-two order): {int(math.log2(n))}")
print(f"Primitive n-th root of unity (w): {hex(omega)}, limbs: {omega_limbs}")
print(f"w^n mod q = {pow(omega, n, q)}")

# compute root of unity of order logn=24 in the ring q based on Rou of the prime fields
p1_rou_27 = 0x00000089 # Baby bear
p1_rou_24 = pow(p1_rou_27, 8, p1) #w^8 mod p1 for w or order logn=27
p2_rou_24 = 0x6ac49f88 # Koala bear
q_rou_24 = from_rns(p1, p2, p1_rou_24, p2_rou_24) #crt
print(f"(logn=24) w^n mod q = {pow(q_rou_24, 1<<24, q)}")
print(f"Rou in the ring q = {to_32b_limbs_str(q_rou_24)}")

