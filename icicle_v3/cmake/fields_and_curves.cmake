
# Define available fields with an index and their supported features
# Format: index:field:features
set(ICICLE_FIELDS
  1001:babybear:NTT,EXT_FIELD
  1002:stark252:NTT
)

# Define available curves with an index and their supported features
# Format: index:curve:features
set(ICICLE_CURVES
  1:bn254:NTT,MSM,G2,ECNTT
  2:bls12_381:NTT,MSM,G2,ECNTT
  3:bls12_377:NTT,MSM,G2,ECNTT
  4:bw6_761:NTT,MSM,G2,ECNTT
  5:grumpkin:MSM
)