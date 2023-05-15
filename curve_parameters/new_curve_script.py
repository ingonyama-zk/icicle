import json
import math
import os
from sympy.ntheory import isprime, primitive_root
import subprocess
import random 
import sys

data = None
with open(sys.argv[1]) as json_file:
    data = json.load(json_file)

curve_name = data["curve_name"]
modolus_p = data["modolus_p"]
bit_count_p = data["bit_count_p"]
limb_p =  data["limb_p"]
ntt_size = data["ntt_size"]
modolus_q = data["modolus_q"]
bit_count_q = data["bit_count_q"] 
limb_q = data["limb_q"]
weierstrass_b = data["weierstrass_b"]
gen_x = data["gen_x"]
gen_y = data["gen_y"]


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

def create_field_parameters_struct(modulus, modulus_bits_count,limbs,ntt,size,name):
    s = " struct "+name+"{\n"
    s += "   static constexpr unsigned limbs_count = " + str(limbs)+";\n"
    s += "   static constexpr storage<limbs_count> modulus = {"+to_hex(modulus,8*limbs)[:-2]+"};\n"
    s += "   static constexpr storage<limbs_count> modulus_2 = {"+to_hex(modulus*2,8*limbs)[:-2]+"};\n"   
    s += "   static constexpr storage<limbs_count> modulus_4 = {"+to_hex(modulus*4,8*limbs)[:-2]+"};\n"
    s += "   static constexpr storage<2*limbs_count> modulus_wide = {"+to_hex(modulus,8*limbs*2)[:-2]+"};\n"
    s += "   static constexpr storage<2*limbs_count> modulus_sqared = {"+to_hex(modulus*modulus,8*limbs)[:-2]+"};\n"  
    s += "   static constexpr storage<2*limbs_count> modulus_sqared_2 = {"+to_hex(modulus*modulus*2,8*limbs)[:-2]+"};\n"   
    s += "   static constexpr storage<2*limbs_count> modulus_sqared_4 = {"+to_hex(modulus*modulus*2*2,8*limbs)[:-2]+"};\n"   
    s += "   static constexpr unsigned modulus_bits_count = "+str(modulus_bits_count)+";\n"
    m = int(math.floor(int(pow(2,2*modulus_bits_count) // modulus)))
    s += "   static constexpr storage<limbs_count> m = {"+ to_hex(m,8*limbs)[:-2] +"};\n"
    s += "   static constexpr storage<limbs_count> one = {"+ to_hex(1,8*limbs)[:-2] +"};\n"
    s += "   static constexpr storage<limbs_count> zero = {"+ to_hex(0,8*limbs)[:-2] +"};\n"

    if ntt:
        for k in range(size):
            omega = get_root_of_unity(int(pow(2,k+1)))
            s += "   static constexpr storage<limbs_count> omega"+str(k+1)+"= {"+ to_hex(omega,8*limbs)[:-2]+"};\n"
        for k in range(size):
            omega = get_root_of_unity(int(pow(2,k+1)))
            s += "   static constexpr storage<limbs_count> omega_inv"+str(k+1)+"= {"+ to_hex(pow(omega, -1, modulus),8*limbs)[:-2]+"};\n"
        for k in range(size):
            s += "   static constexpr storage<limbs_count> inv"+str(k+1)+"= {"+ to_hex(pow(int(pow(2,k+1)), -1, modulus),8*limbs)[:-2]+"};\n"  
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


# Create Cuda interface

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
    


# Create Rust interface and tests

if limb_p == limb_q: 
    with open("./src/curve_templates/curve_same_limbs.rs", "r") as curve_file:
        content = curve_file.read()
        content = content.replace("CURVE_NAME_U",curve_name.upper())
        content = content.replace("CURVE_NAME_L",curve_name.lower())
        content = content.replace("_limbs_p",str(limb_p * 8 * 4))
        content = content.replace("limbs_p",str(limb_p))
        text_file = open("./src/curves/"+curve_name+".rs", "w")
        n = text_file.write(content)
        text_file.close()
else:
    with open("./src/curve_templates/curve_different_limbs.rs", "r") as curve_file:
        content = curve_file.read()
        content = content.replace("CURVE_NAME_U",curve_name.upper())
        content = content.replace("CURVE_NAME_L",curve_name.lower())
        content = content.replace("_limbs_p",str(limb_p * 8 * 4))
        content = content.replace("limbs_p",str(limb_p))
        content = content.replace("_limbs_q",str(limb_q * 8 * 4))
        content = content.replace("limbs_q",str(limb_q))
        text_file = open("./src/curves/"+curve_name+".rs", "w")
        n = text_file.write(content)
        text_file.close()

with open("./src/curve_templates/test.rs", "r") as test_file:
    content = test_file.read()
    content = content.replace("CURVE_NAME_U",curve_name.upper())
    content = content.replace("CURVE_NAME_L",curve_name.lower())
    text_file = open("./src/test_"+curve_name+".rs", "w")
    n = text_file.write(content)
    text_file.close()
    
with open('./src/curves/mod.rs', 'a') as f:
    f.write('\n pub mod ' + curve_name + ';')

with open('./src/lib.rs', 'a') as f:
    f.write('\npub mod ' + curve_name + ';')