import math
import os
from sympy.ntheory import isprime, primitive_root
import subprocess
import random 
import sys

curve_name = "bls12_381"
modolus_p = 52435875175126190479447740508185965837690552500527637822603658699938581184513
bit_count_p = 255
limb_p =  8
ntt_size = 32
modolus_q = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
bit_count_q = 381 
limb_q = 12
weierstrass_b = 4
gen_x = 3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507
gen_y = 1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569


# curve_name = "bls12_377"
# modolus_p = 8444461749428370424248824938781546531375899335154063827935233455917409239041
# bit_count_p = 253
# limb_p =  8
# ntt_size = 32
# modolus_q = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
# bit_count_q = 377 
# limb_q = 12
# weierstrass_b = 1
# gen_x = 81937999373150964239938255573465948239988671502647976594219695644855304257327692006745978603320413799295628339695
# gen_y = 241266749859715473739788878240585681733927191168601896383759122102112907357779751001206799952863815012735208165030


# curve_name = "bn254"
# modolus_p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
# bit_count_p = 254
# limb_p =  8
# ntt_size = 16
# modolus_q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
# bit_count_q = 254 
# limb_q = 8
# weierstrass_b = 3
# gen_x = 1
# gen_y = 2


def to_hex(val, length):
    x = str(hex(val))[2:]
    if len(x) % 8 != 0:
        x = "0" * (8-len(x) % 8) + x
    if len(x) != length:
        x = "0" * (length-len(x)) + x
    n = 8
    chunks = [x[i:i+n] for i in range(0, len(x), n)][::-1]
    s = ""
    for c in chunks:
        s += "0x" + c + ", "
    return s


def get_root_of_unity(order: int) -> int:
    assert (modolus_p - 1) % order == 0
    return pow(5, (modolus_p - 1) // order, modolus_p)

def create_field_parameters_struct(modolus, modulus_bits_count,limbs,ntt,size,name):
    s = " struct "+name+"{\n"
    s += "   static constexpr unsigned limbs_count = " + str(limbs)+";\n"
    s += "   static constexpr storage<limbs_count> modulus = {"+to_hex(modolus,8*limbs)[:-2]+"};\n"
    s += "   static constexpr storage<limbs_count> modulus_2 = {"+to_hex(modolus*2,8*limbs)[:-2]+"};\n"   
    s += "   static constexpr storage<limbs_count> modulus_4 = {"+to_hex(modolus*4,8*limbs)[:-2]+"};\n"
    s += "   static constexpr storage<2*limbs_count> modulus_wide = {"+to_hex(modolus,8*limbs*2)[:-2]+"};\n"
    s += "   static constexpr storage<2*limbs_count> modulus_sqared = {"+to_hex(modolus*modolus,8*limbs)[:-2]+"};\n"  
    s += "   static constexpr storage<2*limbs_count> modulus_sqared_2 = {"+to_hex(modolus*modolus*2,8*limbs)[:-2]+"};\n"   
    s += "   static constexpr storage<2*limbs_count> modulus_sqared_4 = {"+to_hex(modolus*modolus*2*2,8*limbs)[:-2]+"};\n"   
    s += "   static constexpr unsigned modulus_bits_count = "+str(modulus_bits_count)+";\n"
    m = int(math.floor(int(pow(2,2*modulus_bits_count) // modolus)))
    s += "   static constexpr storage<limbs_count> m = {"+ to_hex(m,8*limbs)[:-2] +"};\n"
    s += "   static constexpr storage<limbs_count> one = {"+ to_hex(1,8*limbs)[:-2] +"};\n"
    s += "   static constexpr storage<limbs_count> zero = {"+ to_hex(0,8*limbs)[:-2] +"};\n"

    if ntt:
        for k in range(size):
            omega = get_root_of_unity(int(pow(2,k+1)))
            s += "   static constexpr storage<limbs_count> omega"+str(k+1)+"= {"+ to_hex(omega,8*limbs)[:-2]+"};\n"
        for k in range(size):
            omega = get_root_of_unity(int(pow(2,k+1)))
            s += "   static constexpr storage<limbs_count> omega_inv"+str(k+1)+"= {"+ to_hex(pow(omega, -1, modolus),8*limbs)[:-2]+"};\n"
        for k in range(size):
            s += "   static constexpr storage<limbs_count> inv"+str(k+1)+"= {"+ to_hex(pow(int(pow(2,k+1)), -1, modolus),8*limbs)[:-2]+"};\n"  
    s+=" };\n"   
    return s

def create_gen():
    s = " struct group_generator {\n"
    s += "  static constexpr storage<fq_config::limbs_count> generator_x = {"+to_hex(gen_x,8*limb_q)[:-2]+ "};\n"
    s += "  static constexpr storage<fq_config::limbs_count> generator_y = {"+to_hex(gen_y,8*limb_q)[:-2]+ "};\n"
    s+=" };\n" 
    return s

def get_config_file_content(modolus_p, bit_count_p, limb_p, ntt_size, modolus_q, bit_count_q, limb_q, weierstrass_b):
    file_content = ""
    file_content += "#pragma once\n#include \"../../utils/storage.cuh\"\n"
    file_content += "namespace PARAMS_"+curve_name.upper()+"{\n"
    file_content += create_field_parameters_struct(modolus_p,bit_count_p,limb_p,True,ntt_size,"fp_config")
    file_content += create_field_parameters_struct(modolus_q,bit_count_q,limb_q,False,0,"fq_config")
    file_content += " static constexpr unsigned weierstrass_b = " + str(weierstrass_b)+ ";\n"
    file_content += create_gen()
    file_content+="}\n"
    return file_content


newpath = "./icicle/curves/"+curve_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)

fc = get_config_file_content(modolus_p, bit_count_p, limb_p, ntt_size, modolus_q, bit_count_q, limb_q, weierstrass_b)
text_file = open("./icicle/curves/"+curve_name+"/params.cuh", "w")
n = text_file.write(fc)
text_file.close()

with open("./icicle/curves/curve_template/lde.cu", "r") as lde_file:
    content = lde_file.read()
    content = content.replace("CURVE_NAME_U",curve_name.upper())
    content = content.replace("CURVE_NAME_L",curve_name.lower())
    text_file = open("./icicle/curves/"+curve_name+"/lde.cu", "w")
    n = text_file.write(content)
    text_file.close()
    
with open("./icicle/curves/curve_template/msm.cu", "r") as msm_file:
    content = msm_file.read()
    content = content.replace("CURVE_NAME_U",curve_name.upper())
    content = content.replace("CURVE_NAME_L",curve_name.lower())
    text_file = open("./icicle/curves/"+curve_name+"/msm.cu", "w")
    n = text_file.write(content)
    text_file.close()

with open("./icicle/curves/curve_template/ve_mod_mult.cu", "r") as ve_mod_mult_file:
    content = ve_mod_mult_file.read()
    content = content.replace("CURVE_NAME_U",curve_name.upper())
    content = content.replace("CURVE_NAME_L",curve_name.lower())
    text_file = open("./icicle/curves/"+curve_name+"/ve_mod_mult.cu", "w")
    n = text_file.write(content)
    text_file.close()
    

namespace = '#include "params.cuh"\n'+'''namespace CURVE_NAME_U {
    typedef Field<PARAMS_CURVE_NAME_U::fp_config> scalar_field_t;\
    typedef scalar_field_t scalar_t;\
    typedef Field<PARAMS_CURVE_NAME_U::fq_config> point_field_t;
    typedef Projective<point_field_t, scalar_field_t, PARAMS_CURVE_NAME_U::group_generator, PARAMS_CURVE_NAME_U::weierstrass_b> projective_t;
    typedef Affine<point_field_t> affine_t;
}'''

with open('./icicle/curves/'+curve_name+'/curve_config.cuh', 'w') as f:
    f.write(namespace.replace("CURVE_NAME_U",curve_name.upper()))
    
    
eq = '''
#include <cuda.h>\n
#include "curve_config.cuh"\n
#include "../../primitives/projective.cuh"\n
extern "C" bool eq_CURVE_NAME_L(CURVE_NAME_U::projective_t *point1, CURVE_NAME_U::projective_t *point2, size_t device_id = 0)
{
    return (*point1 == *point2);
}'''

with open('./icicle/curves/'+curve_name+'/projective.cu', 'w') as f:
    f.write(eq.replace("CURVE_NAME_U",curve_name.upper()).replace("CURVE_NAME_L",curve_name.lower()))

supported_operations = '''
#include "projective.cu"
#include "lde.cu"
#include "msm.cu"
#include "ve_mod_mult.cu"
'''

with open('./icicle/curves/'+curve_name+'/supported_operations.cu', 'w') as f:
    f.write(supported_operations.replace("CURVE_NAME_U",curve_name.upper()).replace("CURVE_NAME_L",curve_name.lower()))
    
with open('./icicle/curves/index.cu', 'a') as f:
    f.write('\n#include "'+curve_name.lower()+'/supported_operations.cu"')