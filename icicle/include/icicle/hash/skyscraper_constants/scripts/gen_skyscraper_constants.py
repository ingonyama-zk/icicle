import subprocess

fields = [
    ("bn254", "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001", 18, [1, 2, 3]),
    ("bls12_381", "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 18, [1, 2, 3]),
    # Add more fields as needed
]

for name, modulus, R, n_list in fields:
    n_str = ",".join(str(n) for n in n_list)
    print(f"Generating Skyscraper constants for {name}, n={n_list}, R={R}")
    subprocess.run([
        "sage", "skyscraper_params.sage", name, modulus, str(R), n_str
    ], check=True) 