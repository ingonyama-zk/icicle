package config

type FieldData struct {
	PackageName string
	Field string
	LimbsNum int
	SupportsExtension bool
	ExtensionLimbsNum int
	SupportsNTT bool
	SupportsPoseidon bool
	GnarkImport string
}

type HashData struct {
	PackageName string
	Hash string
}

type CurveData struct {
	PackageName    string
	Curve          string
	GnarkImport    string
	SupportsECNTT bool
	ScalarFieldData FieldData
	BaseFieldData FieldData
	G2LimbsNum int
}

var Curves []CurveData
var Fields []FieldData
var Hashes []HashData

func addCurve(curve CurveData) {
	Curves = append(Curves, curve)
}

func addField(field FieldData) {
	Fields = append(Fields, field)
}

func addHash(hash HashData) {
	Hashes = append(Hashes, hash)
}
