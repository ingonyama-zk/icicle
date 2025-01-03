#!/usr/bin/env python3
"""
This script generates extern declarators for every curve/field
"""

from itertools import chain
from pathlib import Path
from string import Template

API_PATH = Path(__file__).resolve().parent.parent.joinpath("icicle").joinpath("include").joinpath("icicle").joinpath("api")
TEMPLATES_PATH = API_PATH.joinpath("templates")

"""
Defines a set of curves to generate API for.
A set corresponding to each curve contains headers that shouldn't be included.
"""
CURVES_CONFIG = {
    "bn254": [
        "field_ext.template",
        "vec_ops_ext.template",
        "ntt_ext.template",
    ],
    "bls12_381": [
        # "poseidon2.template",
        "field_ext.template",
        "vec_ops_ext.template",
        "ntt_ext.template",
    ],
    "bls12_377": [
        # "poseidon2.template",
        "field_ext.template",
        "vec_ops_ext.template",
        "ntt_ext.template",
    ],
    "bw6_761": [
        # "poseidon2.template",
        "field_ext.template",
        "vec_ops_ext.template",
        "ntt_ext.template",
    ],
    "grumpkin": {
        # "poseidon2.template",
        "curve_g2.template",
        "msm_g2.template",
        "ecntt.template",
        "ntt.template",
        "vec_ops_ext.template",
        "field_ext.template",
        "ntt_ext.template",
    },
}

"""
Defines a set of fields to generate API for.
A set corresponding to each field contains headers that shouldn't be included.
"""
FIELDS_CONFIG = {
    "babybear": {
        # "poseidon.template",
    },
    "stark252": {
        # "poseidon.template",
        # "poseidon2.template",
        "field_ext.template",
        "vec_ops_ext.template",
        "ntt_ext.template",
    },
    "koalabear": {
    }
    # "m31": {
    #     "ntt_ext.template",
    #     "ntt.template",
    #     "poseidon.template",
    #     "poseidon2.template",
    # }
}

COMMON_INCLUDES = []

WARN_TEXT = """\
// WARNING: This file is auto-generated by a script.
// Any changes made to this file may be overwritten.
// Please modify the code generation script instead.
// Path to the code generation script: scripts/gen_c_api.py

"""

INCLUDE_ONCE = """\
#pragma once

"""

CURVE_HEADERS = list(TEMPLATES_PATH.joinpath("curves").iterdir())
FIELD_HEADERS = list(TEMPLATES_PATH.joinpath("fields").iterdir())

if __name__ == "__main__":

    # Generate API for ingo_curve
    for curve, skip in CURVES_CONFIG.items():
        curve_api = API_PATH.joinpath(f"{curve}.h")

        headers = [header for header in chain(CURVE_HEADERS, FIELD_HEADERS) if header.name not in skip]
        
        # Collect includes
        includes = COMMON_INCLUDES.copy()
        includes.append(f'#include "icicle/curves/params/{curve}.h"')
        if any(header.name.startswith("ntt") for header in headers):
            includes.append('#include "icicle/ntt.h"')
        if any(header.name.startswith("msm") for header in headers):
            includes.append('#include "icicle/msm.h"')
        if any(header.name.startswith("vec_ops") for header in headers):
            includes.append('#include "icicle/vec_ops.h"')
        if any(header.name.startswith("poseidon.h") for header in headers):
            includes.append('#include "poseidon/poseidon.h"')
        if any(header.name.startswith("poseidon2.h") for header in headers):
            includes.append('#include "poseidon2/poseidon2.h"')

        contents = WARN_TEXT + INCLUDE_ONCE.format(curve.upper()) + "\n".join(includes) + "\n\n"
        for header in headers:
            with open(header) as f:
                template = Template(f.read())
            contents += template.safe_substitute({
                "CURVE": curve,
                "FIELD": curve,
            })
            contents += "\n\n"        

        with open(curve_api, "w") as f:
            f.write(contents)


    # Generate API for ingo_field
    for field, skip in FIELDS_CONFIG.items():
        field_api = API_PATH.joinpath(f"{field}.h")

        headers = [header for header in FIELD_HEADERS if header.name not in skip]
        
        # Collect includes
        includes = COMMON_INCLUDES.copy()
        includes.append(f'#include "icicle/fields/stark_fields/{field}.h"')
        if any(header.name.startswith("ntt") for header in headers):
            includes.append('#include "icicle/ntt.h"')
        if any(header.name.startswith("vec_ops") for header in headers):
            includes.append('#include "icicle/vec_ops.h"')
        if any(header.name.startswith("poseidon.h") for header in headers):
            includes.append('#include "icicle/poseidon/poseidon.h"')
        if any(header.name.startswith("poseidon2.h") for header in headers):
            includes.append('#include "icicle/poseidon2/poseidon2.h"')

        contents = WARN_TEXT + INCLUDE_ONCE.format(field.upper()) + "\n".join(includes) + "\n\n"
        for header in headers:
            with open(header) as f:
                template = Template(f.read())
            contents += template.safe_substitute({
                "FIELD": field,
            })
            contents += "\n\n"        

        with open(field_api, "w") as f:
            f.write(contents)