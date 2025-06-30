# Close match: with manual pre/post twist
import numpy as np
import unittest
from math import ceil


def balance(x, q):
    reduced= x % q
    """Convert x in [0, q) to balanced representation (-q/2, q/2]"""
    return reduced - q if reduced >= (q // 2) else reduced

def operator_norm(poly, N, q):
    N = len(poly)
    # Convert to balanced form: (-q/2, q/2]
    balanced = np.array([balance(x,q) for x in poly], dtype=np.float32)
    psi = np.exp(1j * np.pi / N)
    indices = np.arange(N)
    twist = psi ** indices
    twisted = twist * np.array(balanced)
    fft_result = np.fft.fft(twisted)
    return np.max(np.abs(fft_result))
    # twist is redundant when compute norm
    # untwisted = fft_result * (psi ** -indices)
    # return np.max(np.abs(untwisted))

# --- Tests ---
class TestOperatorNorm(unittest.TestCase):

    def setUp(self):
        self.N = 64
        self.q = 2**62 - 57

    def test_simple_polynomial(self):
        poly = [i * 100 for i in range(self.N)]
        norm = operator_norm(poly, self.N, self.q)
        print("norm (simple):", norm)

    def test_zero_polynomial(self):
        poly = [0] * self.N
        norm = operator_norm(poly, self.N, self.q)
        self.assertEqual(norm, 0.0)
        
    def test_alternating_values(self):
        # Alternating large/small pattern
        poly = [(i % 2) * 5000 for i in range(self.N)]
        norm = operator_norm(poly, self.N, self.q)
        print("norm (alternating):", norm)
        
    def test_small_negative_values(self):
        # Test for a(x) = (q - 2) * X
        poly = [0] * self.N
        poly[1] = self.q - 2
        norm = operator_norm(poly, self.N, self.q)
        self.assertLess(norm, 2.1) 
        print("norm (-2x):", norm)

if __name__ == "__main__":
    unittest.main()