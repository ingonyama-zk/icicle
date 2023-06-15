package config

type Curve struct {
	CurveName   string
	PackageName string
	ScalarSize  int
	BaseSize    int
}

var BN_254 = Curve{
	CurveName:   "BN254",
	PackageName: "bn254",
	ScalarSize:  8,
	BaseSize:    8,
}

var BLS_12_377 = Curve{
	CurveName:   "BLS12377",
	PackageName: "bls12377",
	ScalarSize:  8,
	BaseSize:    12,
}
