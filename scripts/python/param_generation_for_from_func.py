'''
This script is used to generate the required parameters by the from(bytes) function in the field class which are:
- unsigned reduced_digits_count
- storage_array<reduced_digits_count, limb_count> reduced_digits
- storage_array<mod_subs_count, 2 * limbs_count + 2> mod_subs
'''

import random

def num_to_limbs(x, limb_bitsize, nof_limbs):
    '''
    This function receives a number x, limb_bitsize - the length of the limb (usually 32 or 64) and the number of limbs (nof_limbs).
    It prints the number in the limbs representation that enables a simple copy-paste into Icicle.
    '''
    stf = "0"*(nof_limbs*limb_bitsize//4)
    stx = str(hex(x))
    stx = stx[2:]
    stf = stf[:-len(stx)] + stx
    stx = stf
    # print(len(stf))
    print("{",end="")
    for i in range(nof_limbs-1,-1,-1):
        if i>0:
            print("0x"+str(stx[(limb_bitsize//4)*i:(limb_bitsize//4)*i+(limb_bitsize//4)]),end=", ")
            # print("std::byte{0x"+str(stx[(limb_bitsize//4)*i:(limb_bitsize//4)*i+(limb_bitsize//4)])+"}",end=", ")
        else:
            print("0x"+str(stx[(limb_bitsize//4)*i:(limb_bitsize//4)*i+(limb_bitsize//4)]), end="}\n")
            # print("std::byte{0x"+str(stx[(limb_bitsize//4)*i:(limb_bitsize//4)*i+(limb_bitsize//4)])+"}", end="}\n")

def limbs_to_num(l, limb_size, nof_limbs):
    '''
    This function receives an array of limbs - l, limb_bitsize - the length of the limb (usually 32 or 64) and the number of limbs (nof_limbs).
    It returns the number represented by the limb list.
    '''
    res = 0
    for i in range(nof_limbs):
        res += (l[i]<<(limb_size*i))
    return res

'''
The way to use the script is to set tlc and p and run.
'''

tlc = 2 # set to number of limbs
p = limbs_to_num([0x00000001, 0xffffffff], 32,tlc) # set to the correct modulus
l = tlc*32 # total bits in storage
m = 576 # max supported bits as input to the storage function
n = (m+l-1)//l # this is the reduced_digits_count - the number of reduces digits required
D = 2**l

# this is a reference testing the algorithm for the from function:
a = random.randint(2**l*n - 200, 2**l*n)
a_temp = a
ai = []
pi = []
s = 0
for i in range(n):
    ai.append(a_temp % D)
    a_temp = a_temp // D
    pi.append(D**i % p)
    # print("xi")
    # num_to_limbs(ai[i],32,18)
    # print("pi")
    # num_to_limbs(pi[i], 32, 18)
    s += ai[i] * pi[i]
    # s = s % p**2
    if s > p**2:
        s = s-p**2
    # print("mul")
    # num_to_limbs(ai[i] * pi[i], 32, 18)
    # print("add")
    # num_to_limbs(s, 32, 18)
# print(s)
print(hex(a%p))
print(hex(s%p))
assert(a%p == s%p) # verifing that the algorithm works

# unsigned reduced_digits_count
print("reduced_digits_count:")
print(n)

# This prints the parameters for storage_array<reduced_digits_count, limb_count> reduced_digits
print("mod_subs:")
for pp in pi:
    num_to_limbs(pp, 32, l//32)

p_sqr = p**2
p_bits = 2*p.bit_length()
max_s = n<<(l+1-p.bit_length())
lu = [0]
# This calculates and prints the parameters for storage_array<mod_subs_count, 2 * limbs_count + 2> mod_subs
print("mod_subs:")
for s in range(1,max_s):
    a = (s<<(p_bits-1))//p
    lu.append(a*p)
    num_to_limbs(a*p,32, 2*tlc+2)