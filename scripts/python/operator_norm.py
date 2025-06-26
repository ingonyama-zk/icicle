# Close match: with manual pre/post twist
import numpy as np
import unittest
from math import ceil

def operator_norm(poly, N, q):
    N = len(poly)
    psi = np.exp(1j * np.pi / N)
    indices = np.arange(N)
    twist = psi ** indices
    twisted = twist * (np.array(poly) % q)
    fft_result = np.fft.fft(twisted)
    untwisted = fft_result * (psi ** -indices)
    return np.max(np.abs(untwisted))

# --- Tests ---
class TestOperatorNorm(unittest.TestCase):

    def setUp(self):
        self.N = 64
        self.q = 2**62 - 57

    def test_simple_polynomial(self):
        poly = [i * 100 for i in range(self.N)]
        norm = operator_norm(poly, self.N, self.q)
        self.assertGreater(norm, 0.0)
        self.assertLess(norm, 1e10)  # loose bound
        print("norm (simple):", norm)

    def test_zero_polynomial(self):
        poly = [0] * self.N
        norm = operator_norm(poly, self.N, self.q)
        self.assertEqual(norm, 0.0)
        
    def test_alternating_values(self):
        # Alternating large/small pattern
        poly = [(i % 2) * 5000 for i in range(self.N)]
        norm = operator_norm(poly, self.N, self.q)
        self.assertGreater(norm, 0.0)
        self.assertLess(norm, 1e10)  # adjust if needed
        print("norm (alternating):", norm)

if __name__ == "__main__":
    unittest.main()