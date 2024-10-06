
# Define available fields with an index and their supported features
# Format: index:field:features
set(ICICLE_FIELDS
  1001:babybear:NTT,EXT_FIELD,POSEIDON
  1002:stark252:NTT,POSEIDON
  1003:m31:EXT_FIELD,POSEIDON
)

# Define available curves with an index and their supported features
# Format: index:curve:features
set(ICICLE_CURVES
  1:bn254:NTT,MSM,G2,ECNTT,POSEIDON
  2:bls12_381:NTT,MSM,G2,ECNTT,POSEIDON
  3:bls12_377:NTT,MSM,G2,ECNTT,POSEIDON
  4:bw6_761:NTT,MSM,G2,ECNTT,POSEIDON
  5:grumpkin:MSM,POSEIDON
)