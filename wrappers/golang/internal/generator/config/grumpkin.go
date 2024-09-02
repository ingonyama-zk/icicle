package config

func init() {
	var grumpkin = CurveData{
		PackageName:         "grumpkin",
		Curve:               "grumpkin",
		GnarkImport:         "",
		SupportsPoly:        false,
		SupportsPoseidon:    true,
		SupportsNTT:         false,
		SupportsECNTT:       false,
		SupportsG2:          false,
		ScalarFieldNumLimbs: 8,
		BaseFieldNumLimbs:   8,
		G2FieldNumLimbs:     0,
	}

	addCurve(grumpkin)
}
