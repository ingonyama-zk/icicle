

class ZqElement:
    def __init__(self, value, q):
        self.value = value % q
        self.q = q
        
    def q(self):
        return self.q

    def __add__(self, other):
        if isinstance(other, ZqElement):
            assert self.q == other.q, "Mismatched moduli"
            return ZqElement(self.value + other.value, self.q)
        return ZqElement(self.value + other, self.q)

    def __sub__(self, other):
        if isinstance(other, ZqElement):
            assert self.q == other.q, "Mismatched moduli"
            return ZqElement(self.value - other.value, self.q)
        return ZqElement(self.value - other, self.q)

    def __mul__(self, other):
        if isinstance(other, ZqElement):
            assert self.q == other.q, "Mismatched moduli"
            return ZqElement(self.value * other.value, self.q)
        return ZqElement(self.value * other, self.q)

    def __repr__(self):
        return f"{self.value} (mod {self.q})"
    

class Labrador(ZqElement):
    q = 0x3b880000f7000001
    def __init__(self, value):
        super().__init__(value, self.q)
        
def calc_nof_digits(q: int, base: int) -> int:
    from math import ceil, log
    nof_digits = ceil(log(q)/ log(base)) + 1 # +1 ??
    # nof_digits = ceil(log(q)/ log(base))
    return nof_digits

# Decompose a Labrador element in a given base using integer digits
def decompose(num: Labrador, base: int) -> list[int]:
    assert base > 1 and base < 2**32
    
    
    q = Labrador.q
    val = num.value
    if base>2 and val > q//2: # for base=2 no negative numbers
        print("Warning: value is greater than q//2")
        val = val - q
    nof_digits = calc_nof_digits(q,base)
    
    digits = []
    for i in range(nof_digits):
        digit = val % base
        val = val // base
        # balanced decomposition means that the digits are in the range [-base//2, base//2)
        if digit > base//2:
            digit = digit - base
            val = val + 1
        print(f"{digit=}, {val=} ")
        digits.append(digit)
    assert(val==0), f"Decomposition failed"
    return digits

def recompose(digits: list[int], base: int) -> Labrador:
    assert base > 1 and base < 2**32
    
    assert len(digits) == calc_nof_digits(Labrador.q, base)
    val = 0
    for i in range(len(digits)):
        val += digits[i] * base**i
    return Labrador(val)


if __name__ == "__main__":
    a = Labrador(0x2223f2a9798ca4d3) # (0x00879ad81ff72bde)
    base = 4
    digits = decompose(a, base)
    all_small = [digit <= base//2 and digit > -base//2 for digit in digits]
    print("Decomposition: ", digits)
    assert all(all_small), f"Decomposition failed: {digits}"
    recomposed_a = recompose(digits, base)
    assert a.value == recomposed_a.value
    print("Success!")
    