package config

var bls12381BaseField = FieldData{
	PackageName: "bls12381",
	Field: "bls12381",
	LimbsNum: 12,
	SupportsExtension: true,
	ExtensionLimbsNum: 24,
	SupportsNTT: false,
	SupportsPoseidon: true,
}

var bls12381ScalarField = FieldData{
	PackageName: "bls12381",
	Field: "bls12381",
	LimbsNum: 8,
	SupportsExtension: false,
	ExtensionLimbsNum: -1,
	SupportsNTT: true,
	SupportsPoseidon: true,
}

var bls12381 = CurveData{
	PackageName:    "bls12381",
	Curve:          "bls12_381",
	GnarkImport:    "bls12-381",
	SupportsECNTT: true,
	ScalarFieldData: bls12381ScalarField,
	BaseFieldData: bls12381BaseField,
}

func init() {
	// addCurve(bls12381)
}
