import json
import math
import os
from string import Template
import sys


argv_list = ['thisfile', 'curve_json', 'command']
new_curve_args = dict(zip(argv_list, sys.argv[:len(argv_list)] + [""]*(len(argv_list) - len(sys.argv))))

def to_hex(val: int, length):
    x = hex(val)[2:]
    if len(x) % 8 != 0:
        x = "0" * (8-len(x) % 8) + x
    if len(x) != length:
        x = "0" * (length-len(x)) + x
    n = 8
    chunks = [x[i:i+n] for i in range(0, len(x), n)][::-1]
    s = ""
    for c in chunks:
        s += f'0x{c}, '
        
    return s[:-2]


def compute_values(modulus, modulus_bit_count, limbs):
    limb_size = 8*limbs
    bit_size = 4*limb_size
    modulus_ = to_hex(modulus,limb_size)
    modulus_2 = to_hex(modulus*2,limb_size)
    modulus_4 = to_hex(modulus*4,limb_size)
    modulus_wide = to_hex(modulus,limb_size*2)
    modulus_squared = to_hex(modulus*modulus,limb_size)
    modulus_squared_2 = to_hex(modulus*modulus*2,limb_size)
    modulus_squared_4 = to_hex(modulus*modulus*4,limb_size)
    m_raw = int(math.floor(int(pow(2,2*modulus_bit_count) // modulus)))
    m = to_hex(m_raw,limb_size)
    one = to_hex(1,limb_size)
    zero = to_hex(0,limb_size)
    montgomery_r = to_hex((2 ** bit_size) % modulus, limb_size)
    montgomery_r_inv = to_hex(((modulus+1)//2)**bit_size % modulus, limb_size)

    return (
        modulus_,
        modulus_2,
        modulus_4,
        modulus_wide,
        modulus_squared,
        modulus_squared_2,
        modulus_squared_4,
        m,
        one,
        zero,
        montgomery_r,
        montgomery_r_inv
    )


def get_fq_params(modulus, modulus_bit_count, limbs, g1_gen_x, g1_gen_y, g2_gen_x_re, g2_gen_x_im, g2_gen_y_re, g2_gen_y_im):
    (
        modulus,
        modulus_2,
        modulus_4,
        modulus_wide,
        modulus_squared,
        modulus_squared_2,
        modulus_squared_4,
        m,
        one,
        zero,
        montgomery_r,
        montgomery_r_inv
    ) = compute_values(modulus, modulus_bit_count, limbs)

    limb_size = 8*limbs
    return {
        'fq_modulus': modulus,
        'fq_modulus_2': modulus_2,
        'fq_modulus_4': modulus_4,
        'fq_modulus_wide': modulus_wide,
        'fq_modulus_squared': modulus_squared,
        'fq_modulus_squared_2': modulus_squared_2,
        'fq_modulus_squared_4': modulus_squared_4,
        'fq_m': m,
        'fq_one': one,
        'fq_zero': zero,
        'fq_montgomery_r': montgomery_r,
        'fq_montgomery_r_inv': montgomery_r_inv,
        'fq_gen_x': to_hex(g1_gen_x, limb_size),
        'fq_gen_y': to_hex(g1_gen_y, limb_size),
        'fq_gen_x_re': to_hex(g2_gen_x_re, limb_size),
        'fq_gen_x_im': to_hex(g2_gen_x_im, limb_size),
        'fq_gen_y_re': to_hex(g2_gen_y_re, limb_size),
        'fq_gen_y_im': to_hex(g2_gen_y_im, limb_size)
    }


def get_fp_params(modulus, modulus_bit_count, limbs, root_of_unity, size=0):
    (
        modulus_,
        modulus_2,
        modulus_4,
        modulus_wide,
        modulus_squared,
        modulus_squared_2,
        modulus_squared_4,
        m,
        one,
        zero,
        montgomery_r,
        montgomery_r_inv
    ) = compute_values(modulus, modulus_bit_count, limbs)
    limb_size = 8*limbs
    if size > 0:
        omega = ''
        omega_inv = ''
        inv = ''
        omegas = []
        omegas_inv = []
        for k in range(size):
            if k == 0:
                om = root_of_unity
            else:
                om = pow(om, 2, modulus)
            omegas.append(om)
            omegas_inv.append(pow(om, -1, modulus))
        omegas.reverse()
        omegas_inv.reverse()
        for k in range(size):
            omega += "\n              {"+ to_hex(omegas[k],limb_size)+"}," if k>0 else "      {"+ to_hex(omegas[k],limb_size)+"},"
            omega_inv += "\n              {"+ to_hex(omegas_inv[k],limb_size)+"}," if k>0 else "      {"+ to_hex(omegas_inv[k],limb_size)+"},"
            inv += "\n              {"+ to_hex(pow(int(pow(2,k+1)), -1, modulus),limb_size)+"}," if k>0 else "      {"+ to_hex(pow(int(pow(2,k+1)), -1, modulus),limb_size)+"},"
  
  
    return {
        'fp_modulus': modulus_,
        'fp_modulus_2': modulus_2,
        'fp_modulus_4': modulus_4,
        'fp_modulus_wide': modulus_wide,
        'fp_modulus_squared': modulus_squared,
        'fp_modulus_squared_2': modulus_squared_2,
        'fp_modulus_squared_4': modulus_squared_4,
        'fp_m': m,
        'fp_one': one,
        'fp_zero': zero,
        'fp_montgomery_r': montgomery_r,
        'fp_montgomery_r_inv': montgomery_r_inv,
        'omega': omega[:-1],
        'omega_inv': omega_inv[:-1],
        'inv': inv[:-1],
    }


def get_weier_params(weierstrass_b, weierstrass_b_g2_re, weierstrass_b_g2_im, size):
    
    return {
        'weier_b': to_hex(weierstrass_b, size),
        'weier_b_g2_re': to_hex(weierstrass_b_g2_re, size),
        'weier_b_g2_im': to_hex(weierstrass_b_g2_im, size),
    }


def get_params(config):
    global ntt_size
    curve_name = config["curve_name"]
    modulus_p = config["modulus_p"]
    bit_count_p = config["bit_count_p"]
    limb_p =  config["limb_p"]
    ntt_size = config["ntt_size"]
    modulus_q = config["modulus_q"]
    bit_count_q = config["bit_count_q"] 
    limb_q = config["limb_q"]
    root_of_unity = config["root_of_unity"]
    if root_of_unity == modulus_p:
        sys.exit("Invalid root_of_unity value; please update in curve parameters")

    weierstrass_b = config["weierstrass_b"]
    weierstrass_b_g2_re = config["weierstrass_b_g2_re"]
    weierstrass_b_g2_im = config["weierstrass_b_g2_im"]
    g1_gen_x = config["g1_gen_x"]
    g1_gen_y = config["g1_gen_y"]
    g2_generator_x_re = config["g2_gen_x_re"]
    g2_generator_x_im = config["g2_gen_x_im"]
    g2_generator_y_re = config["g2_gen_y_re"]
    g2_generator_y_im = config["g2_gen_y_im"]

    params = {
        'curve_name_U': curve_name.upper(),
        'fp_num_limbs': limb_p,
        'fq_num_limbs': limb_q,
        'fp_modulus_bit_count': bit_count_p,
        'fq_modulus_bit_count': bit_count_q,
        'num_omegas': ntt_size
    }
    
    fp_params = get_fp_params(modulus_p, bit_count_p, limb_p, root_of_unity, ntt_size)
    fq_params = get_fq_params(modulus_q, bit_count_q, limb_q, g1_gen_x, g1_gen_y, g2_generator_x_re, g2_generator_x_im, g2_generator_y_re, g2_generator_y_im)
    weier_params = get_weier_params(weierstrass_b, weierstrass_b_g2_re, weierstrass_b_g2_im, 8*limb_q)

    return {
        **params,
        **fp_params,
        **fq_params,
        **weier_params
    }


config = None
with open(new_curve_args['curve_json']) as json_file:
    config = json.load(json_file)

curve_name_lower = config["curve_name"].lower()
curve_name_upper = config["curve_name"].upper()
limb_q = config["limb_q"]
limb_p = config["limb_p"]

# Create Cuda interface

newpath = f'./icicle/curves/{curve_name_lower}'
if not os.path.exists(newpath):
    os.makedirs(newpath)

with open("./icicle/curves/curve_template/params.cuh.tmpl", "r") as params_file:
    params_file_template = Template(params_file.read())
    params = get_params(config)
    params_content = params_file_template.safe_substitute(params)
    with open(f'./icicle/curves/{curve_name_lower}/params.cuh', 'w') as f:
        f.write(params_content)

if new_curve_args['command'] != '-update':
    with open("./icicle/curves/curve_template/lde.cu.tmpl", "r") as lde_file:
        template_content = Template(lde_file.read())
        lde_content = template_content.safe_substitute(
            CURVE_NAME_U=curve_name_upper, 
            CURVE_NAME_L=curve_name_lower
        )
        with open(f'./icicle/curves/{curve_name_lower}/lde.cu', 'w') as f:
            f.write(lde_content)
        
    with open("./icicle/curves/curve_template/msm.cu.tmpl", "r") as msm_file:
        template_content = Template(msm_file.read())
        msm_content = template_content.safe_substitute(
            CURVE_NAME_U=curve_name_upper, 
            CURVE_NAME_L=curve_name_lower
        )
        with open(f'./icicle/curves/{curve_name_lower}/msm.cu', 'w') as f:
            f.write(msm_content)

    with open("./icicle/curves/curve_template/ve_mod_mult.cu.tmpl", "r") as ve_mod_mult_file:
        template_content = Template(ve_mod_mult_file.read())
        ve_mod_mult_content = template_content.safe_substitute(
            CURVE_NAME_U=curve_name_upper, 
            CURVE_NAME_L=curve_name_lower
        )
        with open(f'./icicle/curves/{curve_name_lower}/ve_mod_mult.cu', 'w') as f:
            f.write(ve_mod_mult_content)
        

    with open(f'./icicle/curves/curve_template/curve_config.cuh.tmpl', 'r') as cc:
        template_content = Template(cc.read())
        cc_content = template_content.safe_substitute(
            CURVE_NAME_U=curve_name_upper,
        )
        with open(f'./icicle/curves/{curve_name_lower}/curve_config.cuh', 'w') as f:
            f.write(cc_content)
        

    with open(f'./icicle/curves/curve_template/projective.cu.tmpl', 'r') as proj:
        template_content = Template(proj.read())
        proj_content = template_content.safe_substitute(
            CURVE_NAME_U=curve_name_upper, 
            CURVE_NAME_L=curve_name_lower
        )
        with open(f'./icicle/curves/{curve_name_lower}/projective.cu', 'w') as f:
            f.write(proj_content)


    with open(f'./icicle/curves/curve_template/supported_operations.cu.tmpl', 'r') as supp_ops:
        template_content = Template(supp_ops.read())
        supp_ops_content = template_content.safe_substitute()
        with open(f'./icicle/curves/{curve_name_lower}/supported_operations.cu', 'w') as f:
            f.write(supp_ops_content)


    with open('./icicle/curves/index.cu', 'r+') as f:
        index_text = f.read()
        if index_text.find(curve_name_lower) == -1:
            f.write(f'\n#include "{curve_name_lower}/supported_operations.cu"')
        


    # Create Rust interface and tests

    if limb_p == limb_q: 
        with open("./src/curve_templates/curve_same_limbs.rs", "r") as curve_file:
            content = curve_file.read()
            content = content.replace("CURVE_NAME_U",curve_name_upper)
            content = content.replace("CURVE_NAME_L",curve_name_lower)
            content = content.replace("_limbs_p",str(limb_p * 8 * 4))
            content = content.replace("limbs_p",str(limb_p))
            text_file = open("./src/curves/"+curve_name_lower+".rs", "w")
            n = text_file.write(content)
            text_file.close()
    else:
        with open("./src/curve_templates/curve_different_limbs.rs", "r") as curve_file:
            content = curve_file.read()
            content = content.replace("CURVE_NAME_U",curve_name_upper)
            content = content.replace("CURVE_NAME_L",curve_name_lower)
            content = content.replace("_limbs_p",str(limb_p * 8 * 4))
            content = content.replace("limbs_p",str(limb_p))
            content = content.replace("_limbs_q",str(limb_q * 8 * 4))
            content = content.replace("limbs_q",str(limb_q))
            text_file = open("./src/curves/"+curve_name_lower+".rs", "w")
            n = text_file.write(content)
            text_file.close()

    with open("./src/curve_templates/test.rs", "r") as test_file:
        content = test_file.read()
        content = content.replace("CURVE_NAME_U",curve_name_upper)
        content = content.replace("CURVE_NAME_L",curve_name_lower)
        text_file = open("./src/test_"+curve_name_lower+".rs", "w")
        n = text_file.write(content)
        text_file.close()
        
    with open('./src/curves/mod.rs', 'r+') as f:
        mod_text = f.read()
        if mod_text.find(curve_name_lower) == -1:
            f.write('\npub mod ' + curve_name_lower + ';')

    with open('./src/lib.rs', 'r+') as f:
        lib_text = f.read()
        if lib_text.find(curve_name_lower) == -1:
            f.write('\npub mod ' + curve_name_lower + ';')
