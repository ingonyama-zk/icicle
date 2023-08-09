package config

// {{.SharedLib}}
type Curve struct {
	CurveNameUpperCase string
	CurveNameLowerCase string
	SharedLib          string
	ScalarSize         int
	BaseSize           int
}

var BN_254 = Curve{
	CurveNameUpperCase: "BN254",
	CurveNameLowerCase: "bn254",
	SharedLib:          "-lbn254",
	ScalarSize:         8,
	BaseSize:           8,
}

var BLS_12_377 = Curve{
	CurveNameUpperCase: "BLS12_377",
	CurveNameLowerCase: "bls12377",
	SharedLib:          "-lbls12_377",
	ScalarSize:         8,
	BaseSize:           12,
}

var BLS_12_381 = Curve{
	CurveNameUpperCase: "BLS12381",
	CurveNameLowerCase: "bls12381",
	SharedLib:          "-lbls12_381",
	ScalarSize:         8,
	BaseSize:           12,
}
