package config

// {{.SharedLib}}
type Curve struct {
	PackageName        string
	CurveNameUpperCase string
	CurveNameLowerCase string
	SharedLib          string
	ScalarSize         int
	BaseSize           int
	G2ElementSize      int
}

var BN_254 = Curve{
	PackageName:        "bn254",
	CurveNameUpperCase: "BN254",
	CurveNameLowerCase: "bn254",
	SharedLib:          "-lbn254",
	ScalarSize:         8,
	BaseSize:           8,
	G2ElementSize:      4,
}

var BLS_12_377 = Curve{
	PackageName:        "bls12377",
	CurveNameUpperCase: "BLS12_377",
	CurveNameLowerCase: "bls12_377",
	SharedLib:          "-lbls12_377",
	ScalarSize:         8,
	BaseSize:           12,
	G2ElementSize:      6,
}

var BLS_12_381 = Curve{
	PackageName:        "bls12381",
	CurveNameUpperCase: "BLS12_381",
	CurveNameLowerCase: "bls12_381",
	SharedLib:          "-lbls12_381",
	ScalarSize:         8,
	BaseSize:           12,
	G2ElementSize:      6,
}
