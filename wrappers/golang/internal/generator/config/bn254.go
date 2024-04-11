package config

var bn254BaseField = FieldData{
	PackageName: "bn254",
	Field: "bn254",
	LimbsNum: 8,
	SupportsExtension: true,
	ExtensionLimbsNum: 16,
	SupportsNTT: false,
	SupportsPoseidon: true,
}

var bn254ScalarField = FieldData{
	PackageName: "bn254",
	Field: "bn254",
	LimbsNum: 8,
	SupportsExtension: false,
	ExtensionLimbsNum: -1,
	SupportsNTT: true,
	SupportsPoseidon: true,
}

var bn254 = CurveData{
	PackageName:    "bn254",
	Curve:          "bn254",
	GnarkImport:    "bn254",
	SupportsECNTT: true,
	ScalarFieldData: bn254ScalarField,
	BaseFieldData: bn254BaseField,
}

func init() {
	addCurve(bn254)
}
