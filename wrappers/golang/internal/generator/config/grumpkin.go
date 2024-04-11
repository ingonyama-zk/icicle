package config

var grumpkinBaseField = FieldData{
	PackageName: "grumpkin",
	Field: "grumpkin",
	LimbsNum: 8,
	SupportsExtension: true,
	ExtensionLimbsNum: 16,
	SupportsNTT: false,
	SupportsPoseidon: true,
}

var grumpkinScalarField = FieldData{
	PackageName: "grumpkin",
	Field: "grumpkin",
	LimbsNum: 8,
	SupportsExtension: false,
	ExtensionLimbsNum: -1,
	SupportsNTT: false,
	SupportsPoseidon: true,
}

var grumpkin = CurveData{
	PackageName:    "grumpkin",
	Curve:          "grumpkin",
	GnarkImport:    "grumpkin",
	SupportsECNTT: false,
	ScalarFieldData: grumpkinScalarField,
	BaseFieldData: grumpkinBaseField,
}

func init() {
	// addCurve(grumpkin)
}
