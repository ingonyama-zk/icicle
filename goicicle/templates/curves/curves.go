package config

type Curve struct {
	CurveName   string
	PackageName string
	SharedLib   string
	Prefix      string
	ScalarSize  int
	BaseSize    int
}

var BN_254 = Curve{
	CurveName:   "BN254",
	PackageName: "bn254",
	SharedLib:   "-lbn254",
	Prefix:      "bn254",
	ScalarSize:  8,
	BaseSize:    8,
}

var BLS_12_377 = Curve{
	CurveName:   "BLS12377",
	PackageName: "bls12377",
	SharedLib:   "-lbn12_377",
	Prefix:      "bls12_377",
	ScalarSize:  8,
	BaseSize:    12,
}

var BLS_12_381 = Curve{
	CurveName:   "BLS12381",
	PackageName: "bls12381",
	SharedLib:   "-lbn12_381",
	Prefix:      "bls12_381",
	ScalarSize:  8,
	BaseSize:    12,
}
