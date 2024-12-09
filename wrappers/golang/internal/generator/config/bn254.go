package config

func init() {
	var bn254 = CurveData{
		PackageName:         "bn254",
		Curve:               "bn254",
		GnarkImport:         "bn254",
		SupportsNTT:         true,
		SupportsECNTT:       true,
		SupportsG2:          true,
		SupportsPoseidon:    true,
		SupportsPoseidon2:   true,
		ScalarFieldNumLimbs: 8,
		BaseFieldNumLimbs:   8,
		G2FieldNumLimbs:     16,
	}

	addCurve(bn254)
}
