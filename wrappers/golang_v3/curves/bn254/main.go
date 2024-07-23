package bn254

// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle_v3/build -licicle_field_bn254 -licicle_curve_bn254 -lstdc++ -Wl,-rpath=${SRCDIR}/../../../../icicle_v3/build
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build/lib -lingo_curve_bn254 -lingo_field_bn254 -lstdc++ -lm
import "C"
