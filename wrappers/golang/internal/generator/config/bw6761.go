package config

func init() {
	var bw6761 = CurveData{
		PackageName:         "bw6761",
		Curve:               "bw6_761",
		GnarkImport:         "bw6-761",
		SupportsNTT:         true,
		SupportsECNTT:       true,
		SupportsG2:          true,
		ScalarFieldNumLimbs: 12,
		BaseFieldNumLimbs:   24,
		G2FieldNumLimbs:     24,
	}

	addCurve(bw6761)
}
