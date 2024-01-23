### Grumpkin curve
The main feature of this
curve is that it forms a cycle with bn254, i.e. its scalar field and base
field respectively are the base field and scalar field of bn254.

```json
{
    "curve_name" : "grumpkin",
    "modulus_p" : 21888242871839275222246405745257275088696311157297823662689037894645226208583,
    "bit_count_p" : 254,
    "limb_p" :  8,
    "ntt_size" : 28,
    "modulus_q" : 21888242871839275222246405745257275088548364400416034343698204186575808495617,
    "bit_count_q" : 254,
    "limb_q" : 8,
    "weierstrass_b" : 1476341431159447801469771925460324463763056721553502262059448657299560726618,
    "g1_gen_x" : 1,
    "g1_gen_y" : 9363053777743881629597603024654311418217976890726246060552911861324066865624,
    "nonresidue" : -1
}
```

### Notes

- There's no g2 for Grumpkin curve
- Should implement Grumpkin curve for tests, because ark-grumpkin doesn't work with current macroses
