package config

type FieldData struct {
	PackageName       string
	Field             string
	LimbsNum          int
	GnarkImport       string
	SupportsExtension bool
	ExtensionLimbsNum int
	SupportsNTT       bool
	SupportsPoseidon  bool
	SupportsPoseidon2 bool
	ROU               int
}

type CurveData struct {
	PackageName         string
	Curve               string
	GnarkImport         string
	SupportsNTT         bool
	SupportsECNTT       bool
	SupportsG2          bool
	SupportsPoseidon    bool
	SupportsPoseidon2   bool
	ScalarFieldNumLimbs int
	BaseFieldNumLimbs   int
	G2FieldNumLimbs     int
}

var Curves []CurveData
var Fields []FieldData

func addCurve(curve CurveData) {
	Curves = append(Curves, curve)
}

func addField(field FieldData) {
	Fields = append(Fields, field)
}
