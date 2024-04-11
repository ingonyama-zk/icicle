package config

var bw6761BaseField = FieldData{
	PackageName: "bw6761",
	Field: "bw6761",
	LimbsNum: 24,
	SupportsExtension: true,
	ExtensionLimbsNum: 24,
	SupportsNTT: false,
	SupportsPoseidon: true,
}

var bw6761ScalarField = FieldData{
	PackageName: "bw6761",
	Field: "bw6761",
	LimbsNum: 12,
	SupportsExtension: false,
	ExtensionLimbsNum: -1,
	SupportsNTT: true,
	SupportsPoseidon: true,
}

var bw6761 = CurveData{
	PackageName:    "bw6761",
	Curve:          "bw6_761",
	GnarkImport:    "bw6-761",
	SupportsECNTT: true,
	ScalarFieldData: bw6761ScalarField,
	BaseFieldData: bw6761BaseField,
}

func init() {
	// addCurve(bw6761)
}
