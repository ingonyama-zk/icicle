package config

func init() {
	var bls12377 = CurveData{
		PackageName:         "bls12377",
		Curve:               "bls12_377",
		GnarkImport:         "bls12-377",
		SupportsPoly:        true,
		SupportsNTT:         true,
		SupportsECNTT:       true,
		SupportsG2:          true,
		ScalarFieldNumLimbs: 8,
		BaseFieldNumLimbs:   12,
		G2FieldNumLimbs:     24,
	}

	addCurve(bls12377)
}
