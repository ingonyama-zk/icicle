package bn254

// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build/src/curves -lingo_curve_bn254 -lstdc++ -lm
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build/src/fields -lingo_field_bn254
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build/src/hash -lingo_hash
import "C"
