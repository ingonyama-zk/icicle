package config

var bls12377BaseField = FieldData{
	PackageName: "bls12377",
	Field: "bls12377",
	LimbsNum: 12,
	SupportsExtension: true,
	ExtensionLimbsNum: 24,
	SupportsNTT: false,
	SupportsPoseidon: true,
}

var bls12377ScalarField = FieldData{
	PackageName: "bls12377",
	Field: "bls12377",
	LimbsNum: 8,
	SupportsExtension: false,
	ExtensionLimbsNum: -1,
	SupportsNTT: true,
	SupportsPoseidon: true,
}

var bls12377 = CurveData{
	PackageName:    "bls12377",
	Curve:          "bls12_377",
	GnarkImport:    "bls12-377",
	SupportsECNTT: true,
	ScalarFieldData: bls12377ScalarField,
	BaseFieldData: bls12377BaseField,
}

func init() {
	// addCurve(bls12377)
}
