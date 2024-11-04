package config

func init() {
	var bls12381 = CurveData{
		PackageName:         "bls12381",
		Curve:               "bls12_381",
		GnarkImport:         "bls12-381",
		SupportsNTT:         true,
		SupportsECNTT:       true,
		SupportsG2:          true,
		SupportsPoseidon:    true,
		ScalarFieldNumLimbs: 8,
		BaseFieldNumLimbs:   12,
		G2FieldNumLimbs:     24,
	}

	addCurve(bls12381)
}
