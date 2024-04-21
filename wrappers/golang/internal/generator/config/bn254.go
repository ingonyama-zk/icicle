package config

func init() {
	var bn254 = CurveData{
		PackageName:         "bn254",
		Curve:               "bn254",
		GnarkImport:         "bn254",
		SupportsPoseidon:    true,
		SupportsNTT:         true,
		SupportsECNTT:       true,
		SupportsG2:          true,
		ScalarFieldNumLimbs: 8,
		BaseFieldNumLimbs:   8,
		G2FieldNumLimbs:     16,
	}

	addCurve(bn254)
}
