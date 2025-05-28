from hashlib import sha256
SHA256 = lambda x: sha256(x).digest()

p_BLS12_381 = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
p_BLS12_377 = 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001
p_BN_254 = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
p_PALLAS = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
p_VESTA = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001

def get_rcons_int(R,n,endianess='big'):
    '''
    Generate (integer) round constants for Skyscraper.
    R ... Number of Feistel rounds (2 can be ignored, because rcons for first and last round are 0)
    n ... Extension field will we Fp^n
    '''
    sha_input = lambda i: int(i).to_bytes(4, endianess) + b'Skyscraper' + int(0).to_bytes(28-len("Skyscraper"),endianess)
    rcgen = lambda i: int.from_bytes(SHA256(sha_input(i)), endianess)
    
    return [rcgen(r) for r in range((R-2)*n)]


class Bar:
    def __init__(self, p, n=1, beta=5, s=8, debug=False, rep='b'):
        '''
        p ... Prime.
        n ... Power for (extension) field GF(p^n).
        beta ... Modulus for extension field x**n+beta.
        s ... Decomposition size (bits).
        
        We follow the following naming convention for variables:
        x ... Finite (base) field element
        X ... Finite (extension) field element
        z ... Integer representation of x
        Z ... List of integers representing coefficients of X
        '''
        
        # Finite (extension) field.
        self.p = p
        self.n = n
        P.<x> = PolynomialRing(GF(p))
        self.F = GF(self.p) if n == 1 else GF(self.p**n, name='a', modulus=x**n+beta, repr='poly') # q=p^n
        
        # Decomposition. Assumption: All chunks have same size s (bits)
        self.s = s
        self.b = ceil(log(p,2))       # Bitsize of prime.
        self.m = ceil(self.b/self.s)  # Number of chunks of size s.
        self.B = self.m*self.s        # Total Bitsize covered by chunks.
        
        # Rotation.
        assert(self.m >= 2 and self.m % 2 == 0)  # At least 2, otherwise rotation makes no sense.
        self.rot = self.m//2
        
        # Other
        self.debug = debug
        self.rep = rep
        assert(self.rep in ['b','x'])
        
        # BIT MANIPULATION FUNCTIONS
        self.bitset = lambda x, b : ((x >> (b - 1)) > 0)
        self.bitinv = lambda x, s : ((x ^^ (2**s - 1)) & (2**s - 1))
        self.rotl = lambda x, s, b: ((x << b) & (2**s - 1)) | (x >> (s - b)) # Cyclic left rotation of s-bit value x by b bits.
        self.dmsb = lambda x, s, b: (x & ((2**s - 1) >> b)) # Discard b MSBs for x (which is of bitsize s)
        
        # CONVERSION HELPER FUNCTIONS:
        
        # Finite field element <-> Integer
        self.i2f = lambda z : self.F(z)   # z is an integer
        self.f2i = lambda x : Integer(x)  # x is in GF(p), or constant element of GF(p^n)
        
        # Finite (extension) field element <-> List of integers
        self.IL2FFE = lambda Z : self.i2f(Z[0]) if self.n == 1 else self.i2f(Z) # Z is integer list of (integer) coefficients
        self.FFE2IL = lambda X : [self.f2i(x) for x in X.polynomial().list()] + [0]*(self.n - 1 - X.polynomial().degree()) # X is finite (extension) field element
            
        # STRING REPRESENTATION FOR DECOMPOSED VALUES (=CHUNKS) FROM BASE FIELD
        self.l2s = lambda zi: f"{zi:0{self.s if self.rep == 'b' else self.s//4}{self.rep}}" # Single s-bit value
        self.zd2s = lambda zd: [self.l2s(zi) for zi in zd] # decomposition of single element
        self.Zd2s = lambda Zd : [self.zd2s(zd) for zd in Zd] # list of decompositions
        
        # STRING REPRESENTATION OF FINITE (EXTENSION) FIELD ELEMENTS
        self.i2s = lambda z : f"{z:0{self.B if self.rep == 'b' else self.B//4}{self.rep}}" # B-bit integer to string
        self.ffe2s = lambda X : [self.i2s(z) for z in self.FFE2IL(X)] # finite (extension) field element to string
    
    
    def __str__(self):
        return f"BAR over {self.F} ({self.b} bits base field). Decomposition of {self.B} bits into {self.m} parts of size {self.s}. Circular left-rotation by {self.rot}."
    
    
    def decompose(self, z):
        '''
        Decomposition of z into chunks [z1,...,zm], each of size self.s.
        z ... Integer to decompose into m chunks of integer values zi.
        '''
        assert(z in ZZ)
        assert(z < self.p)
        
        z_ = z
        zis = [0]*self.m
        for i in range(self.m):
            zis[self.m-1-i] = z_ & (2**self.s - 1)
            z_ = z_ >> self.s # next chunk
        
        if self.debug:
            print(f"DECOMPOSE: {self.i2s(z)} -> {self.zd2s(zis)}")
        
        assert(all([zi < 2**self.s for zi in zis]))
        assert(int(''.join(self.zd2s(zis)),2 if self.rep=='b' else 16) == z)
        return zis
    
    
    def compose(self, zis):
        '''
        Composition of chunks [z1,...,zm], each of size self.s, into integer z.
        zis ... List of integer values to compose into integer z.
        '''
        assert(all([zi in ZZ for zi in zis]))
        assert(all([zi < 2**self.s for zi in zis]))
        #assert(zis[0] < self.p0)
        
        z = zis[0]
        for zi in zis[1:]:
            z = z << self.s  
            z += zi
        
        if self.debug:
            print(f"COMPOSE: {self.zd2s(zis)} -> {self.i2s(z)} -> {self.i2s(z % self.p)}")
        assert(int(''.join(self.zd2s(zis)),2 if self.rep=='b' else 16) == z)
        
        #assert(z < self.p)
        return z if z < self.p else z - self.p
    

    def T(self, z):
        '''
        Internal S-Box (Chi-function): S: F_2^s -> F_2^s, z |-> y
        z ... Integer of max self.s bits.
        truncate ... If true, result is truncated.
        '''
        assert(z in ZZ)
        assert(z < 2**self.s) # ensure that z in GF(2^s)
        assert(gcd(self.s,3) == 1)
        
        t1 = self.rotl(self.bitinv(z,self.s), self.s, 1)
        t2 = self.rotl(z, self.s, 2)
        t3 = self.rotl(z, self.s, 3)
        y = z ^^ (t1 & t2 & t3)        
        y = self.rotl(y, self.s, 1)
        
        assert(y < 2**self.s) # ensure that y in GF(2^s)
  
        if self.debug:
            print(f"T: F_2^{self.s} -> F_2^{self.s}. S({self.l2s(z)}) = {self.l2s(y)}")
        
        return y
    
     
    def __call__(self, X):
        assert(X in self.F)
        # X ... Element of finite (extension) field, coefficients are list of base field elements.
        # x ... Element of base field (coefficient).
        # Z ... List of integers.
        # z ... Single integer.
        # Zd ... List of decomposed integers (list of lists)
        # zd ... Decomposition of z.
        
        # List of coefficients of the polynomial representation of X, interpreted as integers
        Z = self.FFE2IL(X)
        if self.debug:
            #print(Z)
            print("BAR: F_q -> F_q")
            print(f"BAR({X}) = BAR({Z}) = BAR({self.ffe2s(X)}) = ...")
        
        # Decomposition
        Zd = [self.decompose(z) for z in Z]
        assert(len(Zd) == self.n and all([len(zd) == self.m for zd in Zd]))
        
        # Rotation
        Yd = flatten(Zd)
        Yd = Yd[self.rot:] + Yd[:self.rot]
        
        # Back to list of lists
        Yd = [Yd[i:i + self.m] for i in range(0, self.n * self.m, self.m)]
        
        if self.debug:
            print(f"ROTATE LEFT BY {self.rot}:")
            print(f"\tBefore: {self.Zd2s(Zd)}")
            print(f"\tAfter:  {self.Zd2s(Yd)}")
        
        # Apply S-Box to chunks
        Zd = [[self.T(yd[i]) for i in range(self.m)] for yd in Yd]

        # Composition
        Z = [self.compose(zd) for zd in Zd]
        
        # Transform back to finite (extension) field element
        X_ = self.IL2FFE(Z)

        if self.debug:
            #print(Z)
            print(f"BAR({X}) = {X_}")

        return X_


class Skyscraper:    
    def __init__(self, p, n=1, beta=5, s=8, debug=False, debugBar=False, rep='b', montgomery=True):
        '''
        p ... Prime.
        n ... Power for (extension) field GF(p^n).
        beta ... Modulus for extension field x**n+beta.
        num_B ... Number of consecutive Bars B_i
        num_S ... Number of consecutive Squarings S_i
        N ... N to calculate the number of Feistel rounds: R = num_S + (num_B + sum_S) * N. 
        s ... Decomposition (=chunk) size (bits).
        mont ... Use the Montgomery constant in the squaring function.
        '''
        
        # Finite (extension) field.
        self.p = p
        self.n = n
        P.<x> = PolynomialRing(GF(p))
        self.F = GF(self.p) if n == 1 else GF(self.p**n, name='a', modulus=x**n+beta, repr='poly') # q=p^n
        
        # Montgomery Constant
        machine_bit = ((int(p).bit_length() + 63) // 64) * 64
        self.sigma = 2**machine_bit % p
        self.sigma_inv = pow(self.sigma, -1, p)
        self.mont = montgomery
        
        # Define Bar function
        self.BAR = Bar(p=p, n=n, beta=beta, s=s, debug=debugBar, rep=rep)
        
        # Construction of Feistel
#        self.num_B = num_B
#        self.num_S = num_S
#        self.N = N
        self.R = 18 # total number of Feistel rounds
        self.rcons = self.get_rcons()
        self.debug = debug
        
        # Round functions
        self.B = lambda x, i : self.BAR(x) + self.rcons[i]
        self.S = lambda x, i : (x**2 * self.sigma_inv + self.rcons[i]) if self.mont else (x**2 + self.rcons[i])
        
        if self.debug:
            print(self)
            print(self.BAR)
            #print(f"machine bit: {machine_bit}")
        
    
    def get_rcons(self, random_rcons=False):
        rcons_int = get_rcons_int(self.R,self.n)
        rcons_field = [self.BAR.IL2FFE(rcons_int[i:i + self.n]) for i in range(0, len(rcons_int), self.n)]
        return [self.F.zero()] + rcons_field + [self.F.zero()]
    
    def is_square_round(self, i):
        return i not in [6,7,10,11]
    
    def __str__(self):
        return f"SKYSCRAPER over {self.F}. Feistel rounds: " + ' '.join(['x^2']*self.num_S + (['B']*self.num_B + ['x^2']*self.num_S) * self.N)
    
    
    def __call__(self, state, inv=False):
        '''
        Call skyscraper permutation.
        
        state ... List of 2 elements, each of which is either a finite field element or a list of coefficients
        '''
        assert(len(state) == 2)
        
        if type(state[0]) == type(self.F.zero()):
            xL = state[0]
        elif type(state[0]) == type([]):
            xL = self.BAR.IL2FFE(state[0])
        else:
            assert(False)
        
        if type(state[1]) == type(self.F.zero()):
            xR = state[1]
        elif type(state[1]) == type([]):
            xR = self.BAR.IL2FFE(state[1])
        else:
            assert(False)
        
        #print(f"Perm. Input:  ({xL},{xR})")
        
        # Initial twist for inversion
        if inv:
            rounds = range(self.R-1,-1,-1)
            xL,xR = xR,xL
        else:
            rounds = range(self.R)
        
        # Feistel
        for i in rounds:
            yL = xL
            
            if self.is_square_round(i):
                yR = (xR - self.S(xL,i)) if inv else (xR + self.S(xL,i))
            else:
                yR = (xR - self.B(xL,i)) if inv else (xR + self.B(xL,i))
            
            if self.debug:
                print(f"Round {i+1} out: {(xL,xR)} -> {(yL,yR)}")
            
            xL, xR = yR, yL
        
        # Final twist for inversion
        if inv:
            xL,xR = xR,xL
            
        #print(f"Perm. Output: ({xL},{xR})")
        
        return [xL,xR]
    
    
    def compress(self, message):
        '''
        2*n:n compression
        
        message ... list of 2n (base) field elements
        returns list of n (base) field elements
        '''
        assert(len(message) == 2 * self.n)
        
        print(f"In:  {message}")

        perm_in = [self.BAR.IL2FFE(message[i:i + self.n]) for i in range(0,2*self.n,self.n)]
        perm_out = self.__call__(perm_in)
        
        compressed = self.BAR.FFE2IL(perm_in[0] + perm_out[0])
        print(f"Out: {compressed}")
        
        return compressed


# --------------------------------------------------------------------------    
# INSTANCES
# --------------------------------------------------------------------------    
debug = False
debugBar = False
Sky_BLS381_1 = Skyscraper(p=p_BLS12_381, n=1, debug=debug, debugBar=debugBar, s=8)
Sky_BLS381_2 = Skyscraper(p=p_BLS12_381, n=2, debug=debug, debugBar=debugBar, beta=5, s=8)
Sky_BLS381_3 = Skyscraper(p=p_BLS12_381, n=3, debug=debug, debugBar=debugBar, beta=2, s=8)

Sky_BN254_1 = Skyscraper(p=p_BN_254, debug=debug, debugBar=debugBar, n=1, s=8)
Sky_BN254_2 = Skyscraper(p=p_BN_254, debug=debug, debugBar=debugBar, n=2, beta=5, s=8)
Sky_BN254_3 = Skyscraper(p=p_BN_254, debug=debug, debugBar=debugBar, n=3, beta=3, s=8)

Sky_PALLAS_1 = Skyscraper(p=p_PALLAS, n=1, debug=debug, debugBar=debugBar, s=8)
Sky_PALLAS_2 = Skyscraper(p=p_PALLAS, n=2, debug=debug, debugBar=debugBar, beta=5, s=8)
Sky_PALLAS_3 = Skyscraper(p=p_PALLAS, n=3, debug=debug, debugBar=debugBar, beta=2, s=8)

Sky_VESTA_1 = Skyscraper(p=p_VESTA, n=1, debug=debug, debugBar=debugBar, s=8)
Sky_VESTA_2 = Skyscraper(p=p_VESTA, n=2, debug=debug, debugBar=debugBar, beta=5, s=8)
Sky_VESTA_3 = Skyscraper(p=p_VESTA, n=3, debug=debug, debugBar=debugBar, beta=2, s=8)