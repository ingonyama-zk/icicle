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
	SupportsPoly      bool
	ROU               int
}

type HashData struct {
	PackageName string
	Hash        string
}

// Maybe just put limbs in CurveData and no need for individual Field objects
type CurveData struct {
	PackageName         string
	Curve               string
	GnarkImport         string
	SupportsPoly        bool
	SupportsPoseidon    bool
	SupportsNTT         bool
	SupportsECNTT       bool
	SupportsG2          bool
	ScalarFieldNumLimbs int
	BaseFieldNumLimbs   int
	G2FieldNumLimbs     int
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
