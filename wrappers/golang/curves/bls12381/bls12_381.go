package bls12381

// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build/src/curves -lingo_curve_bls12_381 -lstdc++ -lm
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build/src/fields -lingo_field_bls12_381
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build/src/hash -lingo_hash
import "C"
