package bn254

// #cgo LDFLAGS: -L${SRCDIR}/../../../../../icicle/build/src/curves -lingo_curve_bn254 -lstdc++ -lm
import "C"
import (
	bn254Fields "github.com/ingonyama-zk/icicle/wrappers/golang/temp_golang_structure/fields/bn254"
)

type ScalarField = bn254Fields.ScalarField
type BaseField = bn254Fields.BaseField
